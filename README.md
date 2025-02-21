<h1 align="center">
HyperACL: Andersen-Chung-Lang Clustering on Weighted, Directed, Self-looped Graphs and Hypergraphs
</h1>

This repository holds the code for our research paper [Provably Extending PageRank-based Local Clustering Algorithm to Weighted Directed Graphs with Self-Loops and to Hypergraphs](https://arxiv.org/pdf/2412.03008). In this paper, we propose two algorithms that leverage PageRank to find a compact cluster near the given starting instances/nodes: GeneralACL for graphs and HyperACL for hypergraphs

## 🌈 Features
- 💪**Quadratic Optimal**. Both GeneralACL and HyperACL has been proven to be able to identify a quadratically optimal local cluster with high probability. 
- 🚀**Fast**. Both GeneralACL and HyperACL are fast algorithms. After computing the stationary distribution of random walk (which can be very fast with modern hardware and advanced algorithms), both the algorithms are strongly local, i.e., the runtime can be controlled by the size of the output cluster rather than the size of the graph.
- 🔮 Easy to use. The algorithms are implemented in Python and can be easily used in your own projects by feeding the required data format.


## 📦 Installation and Environment Configuration
One can directly install the package dependencies from PyPI. We recommend to create a new conda environment to prevent conflicts with other packages.
```bash
conda create -n hyperacl python=3.11
```
Install [PyTorch](https://pytorch.org/get-started/locally/) according to your system and CUDA version, then install scipy for sparse matrix computation.
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # just for example. please refer to your system and CUDA version
pip install scipy
```

## 📚 Usage
After installation of environment and dependencies, one can run our example code by
```bash
python main.py
```

This example is to find a local cluster of our curated [DBLP-ML](datasets/dblp_v14_ML.json) dataset. For each round/observation, we randomly choose one author from CMU/MIT/Stanford/UCB and find a local cluster around this author. We evaluate the performance by computing the conductance of the local cluster (the smaller the better) and the F1 Score of the local cluster with respect to all authors in the same institution as the chosen author (the larger the better).

The main.py executes our HyperACL. Before 166, the code loads the data and construct the hypergraph. Line 169-174 computes the stationary distribution of the random walk. Line 251-332 is the core of our HyperACL. The rest of the code is to evaluate the performance.

To use our code on your own dataset and general graphs, you can modify the data loading part. Just make sure before entering the core HyperACL code, you have the following variables ready:
- numnodes: the number of nodes in the graph
- P and PT_tensor: P is the transition matrix of your hypergraph or general graph, PT_tensor can be computed given P in line 157-165.


## 🤗 Cite
If you find this repository useful in your research, please consider citing the following paper:
```bibtex
@article{DBLP:journals/corr/abs-2412-03008,
  author       = {Zihao Li and
                  Dongqi Fu and
                  Hengyu Liu and
                  Jingrui He},
  title        = {Provably Extending PageRank-based Local Clustering Algorithm to Weighted
                  Directed Graphs with Self-Loops and to Hypergraphs},
  journal      = {CoRR},
  volume       = {abs/2412.03008},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2412.03008},
  doi          = {10.48550/ARXIV.2412.03008},
  eprinttype    = {arXiv},
  eprint       = {2412.03008},
  timestamp    = {Mon, 13 Jan 2025 21:28:31 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2412-03008.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```