# Shape As Points for Point Clouds with Visibility Information

This repository contains the implementation of 
Shape As Points for Point Clouds with Visibility Information as
described in the paper 
[Deep Surface Reconstruction for Point Clouds with Visibility Information](https://arxiv.org/abs/2202.01810).

The code is largely based on the [original repository](https://github.com/autonomousvision/shape_as_points).

# Data

The data used in this repository can be downloaded [here](https://github.com/raphaelsulzer/dsrv-data).


# Reconstruction

For reconstructing e.g. the ModelNet10 dataset run

`python generate.py configs/pointcloud/modelnet/config`

where `config` should be replaced with
- `modelnetTR.yaml` for reconstruction from a point cloud (traditional Shape As Points)
- `modelnetSV.yaml` for reconstruction from a point cloud augmented with sensor vectors
- `modelnetAP.yaml` for reconstruction from a point cloud augmented with sensor vectors and auxiliary points





## References

If you find the code or data in this repository useful, 
please consider citing

```bibtex
@misc{sulzer2022deep,
      title={Deep Surface Reconstruction from Point Clouds with Visibility Information}, 
      author={Raphael Sulzer and Loic Landrieu and Alexandre Boulch and Renaud Marlet and Bruno Vallet},
      year={2022},
      eprint={2202.01810},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
```bibtex
@inproceedings{Peng2021SAP,
 author    = {Peng, Songyou and Jiang, Chiyu "Max" and Liao, Yiyi and Niemeyer, Michael and Pollefeys, Marc and Geiger, Andreas},
 title     = {Shape As Points: A Differentiable Poisson Solver},
 booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
 year      = {2021}}
```