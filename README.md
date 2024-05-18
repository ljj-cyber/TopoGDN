## TopoGDN

### Project Name

Multivariate Time-Series Anomaly Detection based on Enhancing Graph Attention Networks with Topological Analysis

### Brief Description

This project applies graph attention networks combined with topological analysis to detect anomalies in multivariate time series. It leverages research in topological graph neural networks and graph neural networks to effectively analyze complex time series data.

### Installation Steps

To install this project, you need to install the following Python libraries:

```bash
pip install torch==1.13.1+cu117
pip install torch-cluster==1.6.0+pt113cu116
pip install torch-geometric==1.7.1
pip install torch-persistent-homology==0.1
pip install torch-scatter==2.1.0+pt113cu116
pip install torch-sparse==0.6.15+pt113cu116
pip install torch-spline-conv==1.2.1+pt113cu116
pip install pyg-lib==0.2.0+pt113cu116
```

### How to Use

Run the main program:

```bash
python main.py --dataset msl
```

### Dataset Information

- **MSL Dataset**: The open-source part is readily available for use.
- **SWAT and WADI Datasets**: These can be obtained from [iTrust](https://itrust.sutd.edu.sg/).
- **SMD Dataset**: Please refer to https://github.com/17000cyh/IMDiffusion.

### Acknowledgement

Thanks to the following works for sharing the code repository:

```
@InProceedings{Horn22a,
  author = {Horn, Max and {De Brouwer}, Edward and Moor, Michael and Moreau, Yves and Rieck, Bastian and Borgwardt, Karsten},
  title = {Topological Graph Neural Networks},
  year = {2022},
  booktitle = {International Conference on Learning Representations~(ICLR)},
  url = {https://openreview.net/pdf?id=oxxUMeFwEHd},
}
@inproceedings{deng2021graph,
  title = {Graph neural network-based anomaly detection in multivariate time series},
  author = {Deng, Ailin and Hooi, Bryan},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume = {35},
  number = {5},
  pages = {4027--4035},
  year = {2021}
}
```
