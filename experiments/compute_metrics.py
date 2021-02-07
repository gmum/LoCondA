import argparse
import json
import os
from collections import defaultdict
from pprint import pprint
from os.path import join

import numpy as np
import torch
import torch.utils.data
from datasets.pointflow import (
    ShapeNet15kPointClouds,
    synsetid_to_cate,
)
from losses.pointflow_metrics import EMD_CD, compute_all_metrics
from losses.pointflow_metrics import jsd_between_point_cloud_sets as JSD
from models import aae
from utils.points import generate_points
from utils.util import (
    cuda_setup,
    find_latest_epoch,
    get_weights_dir,
    prepare_results_dir,
    set_seed,
)


def get_test_loader(config):
    dataset = ShapeNet15kPointClouds(
        root_dir=config["data_dir"],
        categories=config["classes"],
        tr_sample_size=config["n_points"],
        te_sample_size=config["n_points"],
        split="val",
        disable_normalization=config["disable_normalization"],
        recenter_per_shape=config["recenter_per_shape"],
        normalize_per_shape=config["normalize_per_shape"],
        normalize_std_per_axis=config["normalize_std_per_axis"],
    )

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    return loader


def evaluate_recon(model, encoder, config, device, output_dir, epoch):
    if "all" in config["classes"] or len(config["classes"]) == 0:
        cates = list(synsetid_to_cate.values())
    else:
        cates = config["classes"]
    all_results = {}
    cate_to_len = {}
    save_dir = output_dir
    transposed = False
    for cate in cates:
        loader = get_test_loader(config)

        all_sample = []
        all_ref = []
        for data in loader:
            tr_pc, te_pc = data["tr_points"], data["te_points"]
            tr_pc = tr_pc.to(device)
            te_pc = te_pc.to(device)
            m, s = data["mean"].float(), data["std"].float()
            m = m.to(device)
            s = s.to(device)

            if tr_pc.size(-1) == 3:
                tr_pc.transpose_(tr_pc.dim() - 2, tr_pc.dim() - 1)
                te_pc.transpose_(te_pc.dim() - 2, te_pc.dim() - 1)
                m.transpose_(m.dim() - 2, m.dim() - 1)
                s.transpose_(s.dim() - 2, s.dim() - 1)
                transposed = True

            _, mu_a, _ = encoder(tr_pc)
            target_networks_weights = model(mu_a)

            out_pc = torch.zeros(tr_pc.shape).to(device)
            for j, target_network_weights in enumerate(
                target_networks_weights
            ):
                target_network = aae.TargetNetwork(
                    config, target_network_weights
                ).to(device)

                target_network_input = generate_points(
                    config=config,
                    epoch=epoch,
                    size=(tr_pc.shape[2], tr_pc.shape[1]),
                )

                out_pc[j] = torch.transpose(
                    target_network(target_network_input.to(device)), 0, 1
                )

            if transposed:
                out_pc = out_pc.transpose(2, 1)
                te_pc = te_pc.transpose(2, 1)

            all_sample.append(out_pc)
            all_ref.append(te_pc)

        sample_pcs = torch.cat(all_sample, dim=0)
        ref_pcs = torch.cat(all_ref, dim=0)

        sample_pcs = sample_pcs.contiguous()
        ref_pcs = ref_pcs.contiguous()

        cate_to_len[cate] = int(sample_pcs.size(0))
        print(
            "Cate=%s Total Sample size:%s Ref size: %s"
            % (cate, sample_pcs.size(), ref_pcs.size())
        )

        # Save it
        np.save(
            os.path.join(save_dir, "%s_out_smp.npy" % cate),
            sample_pcs.cpu().detach().numpy(),
        )
        np.save(
            os.path.join(save_dir, "%s_out_ref.npy" % cate),
            ref_pcs.cpu().detach().numpy(),
        )

        results = EMD_CD(sample_pcs, ref_pcs, 32, accelerated_cd=True)
        results = {
            k: (v.cpu().detach().item() if not isinstance(v, float) else v)
            for k, v in results.items()
        }
        pprint(results)
        all_results[cate] = results

    # Save final results
    print("=" * 80)
    print("All category results:")
    print("=" * 80)
    pprint(all_results)
    save_path = os.path.join(save_dir, "percate_results.npy")
    np.save(save_path, all_results)

    # Compute weighted performance
    ttl_r, ttl_cnt = defaultdict(lambda: 0.0), defaultdict(lambda: 0.0)
    for catename, l in cate_to_len.items():
        for k, v in all_results[catename].items():
            ttl_r[k] += v * float(l)
            ttl_cnt[k] += float(l)
    ttl_res = {k: (float(ttl_r[k]) / float(ttl_cnt[k])) for k in ttl_r.keys()}
    print("=" * 80)
    print("Averaged results:")
    pprint(ttl_res)
    print("=" * 80)

    save_path = os.path.join(save_dir, "results.npy")
    np.save(save_path, all_results)


def evaluate_gen(
    model, config, device, output_dir, epoch, eval_like_in_pointflow
):
    loader = get_test_loader(config)
    all_sample = []
    all_ref = []
    distribution = config["metrics"]["distribution"]
    transposed = False
    dataset: ShapeNet15kPointClouds = loader.dataset

    for data in loader:
        idx, te_pc = data["idx"], data["te_points"]
        te_pc = te_pc.to(device)
        B = te_pc.size(0)
        m, s = data["mean"].float(), data["std"].float()
        m = m.to(device)
        s = s.to(device)
        if te_pc.size(-1) == 3:
            te_pc.transpose_(te_pc.dim() - 2, te_pc.dim() - 1)
            m.transpose_(m.dim() - 2, m.dim() - 1)
            s.transpose_(s.dim() - 2, s.dim() - 1)
            transposed = True

        noise = torch.zeros(B, config["z_size"]).to(device)

        if distribution == "normal":
            noise.normal_(
                config["metrics"]["normal_mu"], config["metrics"]["normal_std"]
            )
        elif distribution == "beta":
            noise_np = np.random.beta(
                config["metrics"]["beta_a"],
                config["metrics"]["beta_b"],
                noise.shape,
            )
            noise = torch.tensor(noise_np).float().round().to(device)

        target_networks_weights = model(noise)

        out_pc = torch.zeros(te_pc.shape).to(device)

        for j, target_network_weights in enumerate(target_networks_weights):
            target_network = aae.TargetNetwork(
                config, target_network_weights
            ).to(device)

            target_network_input = generate_points(
                config=config,
                epoch=epoch,
                size=(te_pc.shape[2], te_pc.shape[1]),
            )

            out_pc[j] = torch.transpose(
                target_network(target_network_input.to(device)), 0, 1
            )

        if transposed:
            out_pc = out_pc.transpose(2, 1)
            te_pc = te_pc.transpose(2, 1)

        if not eval_like_in_pointflow:
            out_pc = dataset.normalize_generated_points(out_pc)
            te_pc = dataset.normalize_given_points(te_pc, idx)
        all_sample.append(out_pc)
        all_ref.append(te_pc)

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)

    sample_pcs = sample_pcs.contiguous()
    ref_pcs = ref_pcs.contiguous()
    print(
        "Generation sample size:%s reference size: %s"
        % (sample_pcs.size(), ref_pcs.size())
    )

    # Save the generative output
    save_dir = output_dir
    np.save(
        os.path.join(save_dir, "model_out_smp.npy"),
        sample_pcs.cpu().detach().numpy(),
    )
    np.save(
        os.path.join(save_dir, "model_out_ref.npy"),
        ref_pcs.cpu().detach().numpy(),
    )

    # Compute metrics
    results = compute_all_metrics(sample_pcs, ref_pcs, 32, accelerated_cd=True)
    results = {
        k: (v.cpu().detach().item() if not isinstance(v, float) else v)
        for k, v in results.items()
    }
    pprint(results)

    sample_pcl_npy = sample_pcs.cpu().detach().numpy()
    ref_pcl_npy = ref_pcs.cpu().detach().numpy()
    jsd = JSD(sample_pcl_npy, ref_pcl_npy)
    print("JSD:%s" % jsd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="File path for evaluation config",
    )
    args = parser.parse_args()

    config = None
    if args.config is not None and args.config.endswith(".json"):
        with open(args.config) as f:
            config = json.load(f)
    assert config is not None

    print("All metrics")
    set_seed(config["seed"])
    weights_path = get_weights_dir(config)
    device = cuda_setup(config["cuda"], config["gpu"])
    print("Finding latest epoch...")
    epoch = find_latest_epoch(weights_path)
    print(f"Epoch: {epoch}")

    distribution = config["metrics"]["distribution"]
    assert distribution in [
        "normal",
        "beta",
    ], "Invalid distribution. Choose normal or beta"

    hyper_network = aae.HyperNetwork(config, device).to(device)
    encoder = aae.Encoder(config).to(device)

    hyper_network.load_state_dict(
        torch.load(join(weights_path, f"{epoch:05}_G.pth"))
    )
    encoder.load_state_dict(
        torch.load(join(weights_path, f"{epoch:05}_E.pth"))
    )

    encoder.eval()
    hyper_network.eval()

    regularization_experiment = (
        "_edge_length_regularization"
        if config["edge_length_regularization"]
        else (
            "_regularize_normal_deviations"
            if config["regularize_normal_deviations"]
            else ""
        )
    )

    results_dir = prepare_results_dir(
        config,
        config["arch"],
        "atlas_test"
        + ("_atlas_net_tn" if config["use_AtlasNet_TN"] else "")
        + regularization_experiment,
        dirs_to_create=[],
    )

    with torch.no_grad():
        # Evaluate reconstruction
        evaluate_recon(
            hyper_network, encoder, config, device, results_dir, epoch
        )
        # Evaluate generation
        evaluate_gen(
            hyper_network,
            config,
            device,
            results_dir,
            epoch,
            config["eval_like_in_pointflow"],
        )


if __name__ == "__main__":
    main()
