<p align="center">
<img src="https://github.com/jungtaekkim/bayeso/blob/main/docs/_static/assets/logo_bayeso_capitalized.svg" width="400" />
</p>

# Batch BayesO: Re-implementation of Batch Bayesian Optimization
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository is to re-implement several baseline methods in batch Bayesian optimization, using [BayesO](https://github.com/jungtaekkim/bayeso).

* [https://bayeso.org](https://bayeso.org)

## Algorithms

* Batch Bayesian optimization with random query selection
* [Batch Bayesian optimization with a kriging believer](https://link.springer.com/chapter/10.1007/978-3-642-10701-6_6)
* [Batch Bayesian optimization with a constant liar](https://link.springer.com/chapter/10.1007/978-3-642-10701-6_6)
* [Batch Bayesian optimization via local penalization](https://arxiv.org/abs/1505.08052)

## Installation

We recommend installing it with `virtualenv`.
You can choose one of the following installation options.

* Using source code (for developer installation)

To install `batch-bayeso` from source code, command it in the `batch-bayeso` root.

```shell
$ pip install .
```

* Using source code (for editable development mode)

To use editable development mode, command it in the `batch-bayeso` root.

```shell
$ pip install -r requirements.txt
$ python setup.py develop
```

* Uninstallation

If you would like to uninstall `batch-bayeso`, command it.

```shell
$ pip uninstall batch-bayeso
```

## Required Packages

Mandatory pacakges are inlcuded in `requirements.txt`.
The following `requirements` files include the package list, the purpose of which is described as follows.

* `requirements-examples.txt`: It needs to be installed to execute the examples included in the `batch-bayeso` repository.

## License
[MIT License](LICENSE)
