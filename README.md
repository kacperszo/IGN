<!-- gnn-benchmark:begin -->
# Running this in gnn-benchmark

Three tiers off one source tree. **Use `ign.torch`** unless you are reproducing a published
number: it is the model reimplemented on plain PyTorch — no DGL, no dgllife, no PyG, no
compiled scatter extension — so it runs on any current torch.

| variant | stack | `gnnb verify` on CASF-2016 |
|---|---|---|
| `ign.reference` | torch 1.3.1, dgl 0.4.3 | 279/279, max abs diff 1e-06 |
| `ign.modern` | torch 2.4.1+cu124, dgl 2.4.0 | 279/279, 0.598 (tolerance 0.75) |
| `ign.torch` | torch 2.5.1+cu124, **no DGL** | 279/279, 0.591 (tolerance 0.75) |

The tolerance is not slack: 264 of 279 complexes reproduce to float noise, and the rest differ
because rdkit 2026 perceives those molecules differently from the 2021 build the checkpoints
were trained against. Hold rdkit fixed and the port is exact — featurise with
`ign_pyg/export_port_graphs.py` inside `ign-ref` and score the arrays with
`ign_pyg/predict.py --graphs`, which gives R = 1.000000, max abs diff 1.5e-05 on all 279.

```bash
podman build --format=docker -f Containerfile.torch -t ign-torch:latest .

gnnb verify --variant ign.torch --dataset data/CASF-2016/coreset
gnnb run --variant ign.torch --capability predict --dataset <complexes> --gpu
gnnb run --variant ign.torch --capability embed   --dataset <complexes>
```

The encoder stops at the pooled graph vector and `FC` is the head — the authors' own boundary,
so nothing is invented. `ign_pyg/test_invariance.py` checks that the prediction does not move
under rotation, translation or a relabelling of the ligand's atoms.

Everything hard-won — why the chirality features are constant zero, which rdkit changes break
what, why `FC` builds three Linear layers when asked for two — is in [CLAUDE.md](CLAUDE.md).

<!-- gnn-benchmark:end -->

---

# InteractionGraphNet(IGN)
a Novel and Efficient Deep Graph Representation Learning Framework for Accurate Protein-Ligand Interaction Predictions.
Accurate quantification of protein-ligand interactions remains a key challenge to structure-based drug design. However, 
traditional machine learning-based methods based on hand-crafted descriptors, one-dimensional protein sequences and/or 
twodimensional graph representations limit their capability to learn the generalized molecular interactions in 3D space. 
Here we proposed a novel deep graph representation learning framework named InteractionGraphNet (IGN) to learn the 
protein-ligand interaction patterns from the 3D structures of protein-ligand complexes in an end-to-end manner. 
In IGN, two independent graph convolution modules were stacked to sequentially learn the intramolecular and 
intermolecular interactions, and only the readouts from the intermolecular convolution module were accepted to force
IGN to capture the protein-ligand interactions in 3D space. Extensive binding affinity prediction, large-scale 
structure-based virtual screening and pose prediction experiments demonstrated that IGN achieved better or competitive 
performance against other state-of-the-art ML-based baselines and docking programs, highlighting the great superiority 
of IGN compared to the other baselines. More importantly, such state-of-the-art performance was proved from the 
successful generalization of truly learning protein-ligand interaction patterns instead of just memorizing certain biased
patterns from proteins or ligands. This source code was tested on the basic environment with `conda==4.5.4` and `cuda==11.0`

![Image text](https://github.com/zjujdj/IGN/blob/master/fig/workflow_new.png)
## Conda Environment Reproduce
Two methods were provided for reproducing the conda environment used in this paper
- **create environment using file packaged by conda-pack**
    
    Download the packaged file [dgl430_v1.tar.gz](https://drive.google.com/file/d/1Rls2ydUSoEjW_rRnvXBzBCcoB4YvcWLQ/view?usp=sharing) 
    and following commands can be used:
    ```python
    mkdir /opt/conda_env/dgl430_v1
    tar -xvf dgl430_v1.tar.gz -C /opt/conda_env/dgl430_v1
    source activate /opt/conda_env/dgl430_v1
    conda-unpack
    ```
  
- **create environment using files provided in `./envs` directory**
    
    The following commands can be used:
    ```python
    conda create --prefix=/opt/conda_env/dgl430_v1 --file conda_packages.txt
    source activate /opt/conda_env/dgl430_v1
    pip install torch==1.3.1+cu92 torchvision==0.4.2+cu92 -f https://download.pytorch.org/whl/torch_stable.html
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    pip install -r pip_packages.txt

    ```
  
## Usage
Users can directly use our well-trained model (depoisted in `./model_save/` directory) to predict the binding affinityies of 
protein-ligand complexes. Other functions including pose prediction, structure-based virtual screening, and 
train the customized binding model is available in the future.
- **step 1: Clone the Repository**
```python
git clone https://github.com/zjujdj/IGN.git
```

- **step 2: Construction of Conda Environment**
```python
# method1 in 'Conda Environment Reproduce' section
mkdir /opt/conda_env/dgl430_v1
tar -xvf dgl430_v1.tar.gz -C /opt/conda_env/dgl430_v1
source activate /opt/conda_env/dgl430_v1
conda-unpack

# method2 in 'Conda Environment Reproduce' section
cd ./IGN/envs
conda create --prefix=/opt/conda_env/dgl430_v1 --file conda_packages.txt
source activate /opt/conda_env/dgl430_v1
pip install torch==1.3.1+cu92 torchvision==0.4.2+cu92 -f https://download.pytorch.org/whl/torch_stable.html
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r pip_packages.txt
```

- **step 3: Binding Affinity Prediction**
```python
cd ./IGN/scripts
python3 model_ign_prediction.py --test_file_path='../input_data/user1'
```

- **step 4: Other functions**
```python
will see you soon
```

## Acknowledgement
Some scripts were based on the [dgl project](https://github.com/awslabs/dgl-lifesci). 
We'd like to show our sincere thanks to them.

