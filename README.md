MR-GNN
======
The source code of ['MR-GNN: Multi-Resolution and Dual Graph Neural Network for Predicting Structured Entity Interactions'](https://arxiv.org/abs/1905.09558?context=cs.LG), IJCAI 2019.

Table of contents
-----------
* [Requirements](https://github.com/prometheusXN/MR-GNN/#requirements)
* [Usage](https://github.com/prometheusXN/MR-GNN/#usage)
* [Reference](https://github.com/prometheusXN/MR-GNN/#reference)

Requirements
-----------
* Python >= 3.5
* [Deepchem-gpu](https://github.com/deepchem/deepchem#requirements) == 2.3.0
* Tensorflow == 1.13.1

Usage
-----------
If you want to repeoduce the experiment result of MR-GNN, please run the code "para_test9_coLSTM.py" and "para_test9_coLSTM_ddi.py".

    pyhthon para_test9_coLSTM.py -g 0

References
-----------

BibTex:

```Latex
@article{xu2019mr,
  title={MR-GNN: Multi-Resolution and Dual Graph Neural Network for Predicting Structured Entity Interactions},
  author={Xu, Nuo and Wang, Pinghui and Chen, Long and Tao, Jing and Zhao, Junzhou},
  journal={Proceedings of IJCAI},
  year={2019}
}
```
