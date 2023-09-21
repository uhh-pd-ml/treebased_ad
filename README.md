# Tree-based Anomaly Detection

Anomaly Detection using tree-based algorithms. This repository contains code to reproduce results of figures 7 and 8 from the paper "Back to the Roots: Tree-Based Algorithms for Weakly Supervised Anomaly Detection" (arXiv link soon). Additionally, this code can be used to run other experiments by adjusting the respective parameters for the training and evaluation of several tree-based algorithms or DNNs.

**IMPORTANT NOTE:** If you `git clone` this repository, make sure to use the `--recursive` option in order to also correctly clone the necessary submodule!
## Requirements

In order to run this code, `git-lfs` is required. Install `git-lfs` with your favourite package manager, or follow the instructions e.g. on [GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)

For other recommended software packages, it is highly recommended to use a virtual environment containing the exact same packages used in our studies for optimal reproducibility. We included an `environment.yml` file which can be used with `anaconda` to install all required packages. For instructions how to install `anaconda`, refer to [this link](https://docs.anaconda.com/free/anaconda/install/index.html). Once `anaconda` is installed, create a new environment from the `environment.yml` file running the following command:

```bash
conda env create -f environment.yml --name <env_name>
```

where `<env_name>` is the chosen name of the new environment.

## Usage

The code can be easily run from a jupyter notebook. The notebook for reproducing results of figures 7 and 8 is called `compare_models.ipynb` and contains step-by-step instructions to either train all the models from scratch or load pre-trained models and evaluate them.

There is also a notebook called `simple_run.ipynb` which can be used to get an idea of the main functions this code provides. It contains a simple example of how to train a model and evaluate it on the test set, as well as plot a SIC curve.

The notebook `tmva_comparison_study.ipynb` contains code for specifically running studies using the ROOT TMVA BDT implementation.

For concrete documentation of the functions, refer to the docstrings in the code. The main code is contained in `utils.py` and the main plotting functions are in `plot_utils.py`.

