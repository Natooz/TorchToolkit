# TorchToolkit

[![PyPI version fury.io](https://badge.fury.io/py/torchtoolkit.svg)](https://pypi.python.org/pypi/torchtoolkit/)

Hi 👋, this is a small Python package containing useful functions to use with PyTorch.
It includes [utilities](torchtoolkit/utils.py), [metrics](torchtoolkit/metrics.py) and [sampling](torchtoolkit/sampling.py) methods to use during and after training a model.

Feel free to use it, take the code for your projects, and raise an issue if you have question or a pull request if you want to contribute.

```shell
pip install torchtoolkit
```
It requires Python 3.8 or above.

Simplest example:

```python
from torchtoolkit.metrics import Accuracy
from torch import randint, randn
from pathlib import Path

acc = Accuracy(mode='top_k', top_kp=5)
for _ in range(10):
    res = randn((16, 32))
    expected = randint(0, 32, (16, ))
    acc(res, expected)  # saving results
acc.save(Path('path', 'to', 'save', 'file.csv'))
acc.analyze()
```

I built it for my own usage, so you won't find documentation besides the docstring.
