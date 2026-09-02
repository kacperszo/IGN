FROM docker.io/mambaorg/micromamba:latest

# conda env: python 3.6 + rdkit + openblas (no MKL — matches original libopenblas-0.3.7)
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && micromamba clean --all --yes

# Pillow, pulled in by matplotlib, links against libtiff.so.5. The base image has no libtiff
# at all and current Debian ships .so.6, so it has to come from conda-forge at the version
# that still provides .so.5. graph_constructor.py does `from pylab import *`, so matplotlib
# is on the import path whether or not anything plots.
RUN micromamba install -y -n base -c conda-forge "libtiff=4.2" && micromamba clean --all --yes

# must appear before any RUN that invokes python
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# build-essential for ProDy C extensions (apt-get requires root)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

# torch 1.3.1 CPU — no cp38 wheel exists; cp36 matches the original env
RUN pip install --no-cache-dir \
    torch==1.3.1+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# DGL 0.4.3 asks which backend to use on first import, interactively. There is no stdin in
# a build, so without this the import check dies on EOFError from input().
ENV DGLBACKEND=pytorch

# DGL CPU pinned before dgllife so dgllife cannot pull a newer version
# (dgl.data.chem.BaseBondFeaturizer was removed in DGL 0.5)
RUN pip install --no-cache-dir dgl==0.4.3

RUN pip install --no-cache-dir \
    dgllife==0.2.3 \
    torchani==2.2 \
    biopython==1.79 \
    scikit-learn==0.23.1 \
    networkx==2.4 \
    h5py==2.10.0 \
    tqdm==4.38.0 \
    llvmlite==0.35.0 \
    numba==0.52.0 \
    ProDy==2.0 \
    prefetch-generator==1.0.1 \
    matplotlib==3.3.3

# smoke-test: verify every import the scripts use at the module level
RUN python -c "from rdkit import Chem; import dgl; from dgl.data.chem import BaseBondFeaturizer; import dgllife; import torch; import torchani; from prody import *; import matplotlib; print('env ok')"

# headless matplotlib — graph_constructor.py does `from pylab import *`
ENV MPLBACKEND=Agg

WORKDIR /work
COPY --chown=$MAMBA_USER:$MAMBA_USER . /work
