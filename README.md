# CCM

Code for replicating the experiments in the paper [Kernel Feature Selection via Conditional Covariance Minimization](https://papers.nips.cc/paper/7270-kernel-feature-selection-via-conditional-covariance-minimization.pdf) by Jianbo Chen\*, Mitchell Stern\*, Martin J. Wainwright, Michael I. Jordan. (\* indicates equal contribution)

## Dependencies
The code for CCM runs with Python and requires Tensorflow of version 1.2.1 or higher. Please `pip install` the following packages:
- `numpy`
- `tensorflow` 

Or you may run the following and in shell to install the required packages:
```shell
git clone https://github.com/Jianbo-Lab/CCM
cd CCM
sudo pip install -r requirements.txt
```

## Running in Docker, MacOS or Ubuntu
We provide as an example the source code to run CCM on the three synthetic datasets in the paper. Run the following commands in shell:

```shell
###############################################
# Omit if already git cloned.
git clone https://github.com/Jianbo-Lab/CCM
cd CCM 
############################################### 
python examples/run_synthetic.py
```

See `core/ccm.py` and `examples/run_synthetic.py` for details. 
## Citation
If you use this code for your research, please cite our [paper](https://papers.nips.cc/paper/7270-kernel-feature-selection-via-conditional-covariance-minimization.pdf):
```
@incollection{NIPS2017_7270,
title = {Kernel Feature Selection via Conditional Covariance Minimization},
author = {Chen, Jianbo and Stern, Mitchell and Wainwright, Martin J and Jordan, Michael I},
booktitle = {Advances in Neural Information Processing Systems 30},
editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
pages = {6949--6958},
year = {2017},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/7270-kernel-feature-selection-via-conditional-covariance-minimization.pdf}
}
```