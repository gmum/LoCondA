import torch
import torch.nn as nn

from utils.util import cuda_setup


class HyperNetwork(nn.Module):
    def __init__(self, config, device, tn_in_features=3, use_atlas_net_tn=False):
        super().__init__()

        self.z_size = config['z_size']
        self.use_bias = config['model']['HN']['use_bias']
        self.relu_slope = config['model']['HN']['relu_slope']
        # target network layers out channels
        if use_atlas_net_tn:
            params = AtlasNetTargetNetwork.count_parameters(config)
        else:
            target_network_out_ch = [tn_in_features] + config['model']['TN']['layer_out_channels'] + [3]
            target_network_use_bias = int(config['model']['TN']['use_bias'])
            params = [(target_network_out_ch[x - 1] + target_network_use_bias) * target_network_out_ch[x]
                      for x in range(1, len(target_network_out_ch))]

        self.model = nn.Sequential(
            nn.Linear(in_features=self.z_size, out_features=64, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=64, out_features=128, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=128, out_features=512, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=512, out_features=1024, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=2048, bias=self.use_bias),
        )

        self.output = [
            nn.Linear(2048, p, bias=True).to(device) for p in params
        ]

        if not config['model']['TN']['freeze_layers_learning']:
            self.output = nn.ModuleList(self.output)

    def forward(self, x):
        output = self.model(x)
        return torch.cat([target_network_layer(output) for target_network_layer in self.output], 1)


class TargetNetwork(nn.Module):
    def __init__(self, config, weights, tn_in_features=3):
        super().__init__()

        self.z_size = config['z_size']
        self.use_bias = config['model']['TN']['use_bias']
        # target network layers out channels
        out_ch = config['model']['TN']['layer_out_channels']

        layer_data, split_index = self._get_layer_data(start_index=0, end_index=out_ch[0] * tn_in_features,
                                                       shape=(out_ch[0], tn_in_features), weights=weights)
        self.layers = {"1": layer_data}

        for x in range(1, len(out_ch)):
            layer_data, split_index = self._get_layer_data(start_index=split_index,
                                                           end_index=split_index + (out_ch[x - 1] * out_ch[x]),
                                                           shape=(out_ch[x], out_ch[x - 1]), weights=weights)
            self.layers[str(x + 1)] = layer_data

        layer_data, split_index = self._get_layer_data(start_index=split_index,
                                                       end_index=split_index + (out_ch[-1] * 3),
                                                       shape=(3, out_ch[-1]), weights=weights)
        self.output = layer_data
        self.activation = torch.nn.ReLU()
        assert split_index == len(weights), 'Incorrect TargetNetwork configuration'

    def forward(self, x):
        for layer_index in self.layers:
            x = torch.mm(x, torch.transpose(self.layers[layer_index]["weight"], 0, 1))
            if self.use_bias:
                assert "bias" in self.layers[layer_index]
                x = x + self.layers[layer_index]["bias"]
            x = self.activation(x)
        return torch.mm(x, torch.transpose(self.output["weight"], 0, 1)) + self.output.get("bias", 0)

    def _get_layer_data(self, start_index, end_index, shape, weights):
        layer_data = {"weight": weights[start_index:end_index].view(shape[0], shape[1])}
        if self.use_bias:
            layer_data["bias"] = weights[end_index:end_index + shape[0]]
            end_index = end_index + shape[0]
        return layer_data, end_index


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.z_size = config['z_size']
        self.use_bias = config['model']['E']['use_bias']
        self.relu_slope = config['model']['E']['relu_slope']

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1, bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, bias=self.use_bias),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 512, bias=True),
            nn.ReLU(inplace=True)
        )

        self.mu_layer = nn.Linear(512, self.z_size, bias=True)
        self.std_layer = nn.Linear(512, self.z_size, bias=True)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        output = self.conv(x)
        output2 = output.max(dim=2)[0]
        logit = self.fc(output2)
        mu = self.mu_layer(logit)
        logvar = self.std_layer(logit)
        z = self.reparameterize(mu, logvar)
        return z, mu, torch.exp(logvar)


# MIT License
# https://github.com/ThibaultGROUEIX/AtlasNet/blob/2baafa5607d6ee3acb5fffaaae100e5efc754f99/model/model_blocks.py#L63
class AtlasNetTargetNetwork(nn.Module):
    """
    Core Atlasnet Function (Mapping2Dto3D) with input size changed from 2 to 5
    Author : Thibault Groueix 01.11.2019
    """
    def __init__(self, config, weights, input_size=5):
        self.bottleneck_size = config['model']['AtlasNet_TN'].get('bottleneck_size', 1024)
        self.input_size = input_size
        self.dim_output = 3
        self.kernel_size = 1
        self.hidden_neurons = config['model']['AtlasNet_TN'].get('hidden_neurons', 512)
        self.num_layers = config['model']['AtlasNet_TN'].get('num_layers', 2)
        self.batch_norm = config['model']['AtlasNet_TN'].get('batch_norm', True)
        self.use_bias = True
        self._device = cuda_setup(config['cuda'], config['gpu'])
        super(AtlasNetTargetNetwork, self).__init__()

        self.layers = []
        layer, split_index = self._get_layer_data(torch.nn.functional.conv1d, start_index=0,
                                                  shape=[self.bottleneck_size, self.input_size, self.kernel_size],
                                                  weights=weights)
        self.layers.append(layer)

        layer, split_index = self._get_layer_data(torch.nn.functional.batch_norm, start_index=split_index,
                                                  shape=[self.bottleneck_size], weights=weights)
        self.layers.append(layer)

        layer, split_index = self._get_layer_data(torch.nn.functional.conv1d, start_index=split_index,
                                                  shape=[self.hidden_neurons, self.bottleneck_size, self.kernel_size],
                                                  weights=weights)
        self.layers.append(layer)

        layer, split_index = self._get_layer_data(torch.nn.functional.batch_norm, start_index=split_index,
                                                  shape=[self.hidden_neurons], weights=weights)
        self.layers.append(layer)

        for _ in range(self.num_layers):
            layer, split_index = self._get_layer_data(torch.nn.functional.conv1d, start_index=split_index,
                                                      shape=[self.hidden_neurons, self.hidden_neurons, self.kernel_size],
                                                      weights=weights)
            self.layers.append(layer)

            layer, split_index = self._get_layer_data(torch.nn.functional.batch_norm, start_index=split_index,
                                                      shape=[self.hidden_neurons], weights=weights)
            self.layers.append(layer)

        layer, split_index = self._get_layer_data(torch.nn.functional.conv1d, start_index=split_index,
                                                  shape=[self.dim_output, self.hidden_neurons, self.kernel_size],
                                                  weights=weights)
        self.layers.append(layer)

        self.activation = torch.nn.ReLU()

        assert split_index == len(weights), 'Incorrect AtlasNet_TN configuration'

    def forward(self, x):
        if x.dim() == 2:
            x = x[None, :, :]
        if x.shape[-2] != self.input_size:
            x = x.transpose(x.dim() - 2, x.dim() - 1)

        for layer in self.layers:
            x = layer['f'](x, *layer['params'])
            if layer['f'] == torch.nn.functional.batch_norm:
                x = self.activation(x)

        return x.transpose(x.dim() - 2, x.dim() - 1)

    def _get_layer_data(self, f, start_index, shape, weights):
        layer_data = {'f': f}
        if f == torch.nn.functional.conv1d:
            end_index = start_index + torch.prod(torch.tensor(shape)).item()
            params = [weights[start_index:end_index].reshape(shape)]
            if self.use_bias:
                start_index = end_index
                end_index += shape[0]
                params.append(weights[start_index:end_index])
            else:
                params.append(None)
        elif f == torch.nn.functional.batch_norm:
            num_features = shape[0]
            end_index = start_index + num_features
            params = [torch.zeros(num_features).to(self._device), torch.ones(num_features).to(self._device),
                      weights[start_index:end_index]]
            if self.use_bias:
                start_index = end_index
                end_index += num_features
                params.append(weights[start_index:end_index])
            else:
                params.append(None)
            params += [self.training or False, 0.1, 1e-5]
        else:
            raise ValueError

        layer_data['params'] = params
        return layer_data, end_index

    @staticmethod
    def count_parameters(config):
        kernel_size = 1
        use_bias = True
        batch_norm = config['model']['AtlasNet_TN'].get('batch_norm', True)
        bottleneck_size = config['model']['AtlasNet_TN'].get('bottleneck_size', 1024)
        hidden_neurons = config['model']['AtlasNet_TN'].get('hidden_neurons', 512)
        num_layers = config['model']['AtlasNet_TN'].get('num_layers', 2)

        conv1 = [(5 * bottleneck_size * kernel_size) + (int(use_bias) * bottleneck_size)] + \
                [2 * bottleneck_size] if batch_norm else []

        conv2 = [(hidden_neurons * bottleneck_size * kernel_size) + (int(use_bias) * hidden_neurons)] + \
                [2 * hidden_neurons] if batch_norm else []

        conv_list = []
        for _ in range(num_layers):
            conv_list += [(hidden_neurons * hidden_neurons * kernel_size) + (int(use_bias) * hidden_neurons)]
            conv_list += [2 * hidden_neurons] if batch_norm else []

        last_conv = [(hidden_neurons * 3 * kernel_size) + (int(use_bias) * 3)]

        return conv1 + conv2 + conv_list + last_conv
