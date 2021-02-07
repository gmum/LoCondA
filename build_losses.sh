#!/bin/bash

export PYTHON_VERSION="$(python -c 'import sys; print("%d.%d" % (sys.version_info[0], sys.version_info[1]))')"

cd losses/pytorch_structural_losses || exit 1
make clean || true
make
