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

```
David Ginsbourger, Rodolphe Le Riche, and Laurent Carraro.
"Kriging is well-suited to parallelize optimization."
Computational intelligence in expensive optimization problems,
pp. 131-162. 2010.
```

* [Batch Bayesian optimization with a constant liar](https://link.springer.com/chapter/10.1007/978-3-642-10701-6_6)

```
David Ginsbourger, Rodolphe Le Riche, and Laurent Carraro.
"Kriging is well-suited to parallelize optimization."
Computational intelligence in expensive optimization problems,
pp. 131-162. 2010.
```

* [Batch Bayesian optimization with pure exploration](https://link.springer.com/chapter/10.1007/978-3-642-40988-2_15)

```
Emile Contal, David Buffoni, Alexandre Robicquet, and Nicolas Vayatis.
"Parallel Gaussian process optimization with upper confidence bound and pure exploration."
In Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases,
pp. 225-240. 2013.
```

* [Batch Bayesian optimization via local penalization](https://proceedings.mlr.press/v51/gonzalez16a.html)

```
Javier Gonzalez, Zhenwen Dai, Philipp Hennig, Neil Lawrence.
"Batch Bayesian optimization via local penalization."
In Proceedings of the International Conference on Artificial Intelligence and Statistics,
pp. 648-657. 2016.
```

## Installation

We recommend installing it with `virtualenv`.
You can choose one of the following installation options.
In addition, you can add `[examples]` in order to install the packages required for the examples included in the `batch-bayeso` repository.
For example, `pip install .[examples]` or `pip install -e .[examples]`.

* Using source code (for developer installation)

To install `batch-bayeso` from source code, command it in the `batch-bayeso` root.

```shell
pip install .
```

* Using source code (for editable development mode)

To use editable development mode, command it in the `batch-bayeso` root.

```shell
pip install -e .
```

* Uninstallation

If you would like to uninstall `batch-bayeso`, command it.

```shell
$ pip uninstall batch-bayeso
```

## License
[MIT License](LICENSE)
