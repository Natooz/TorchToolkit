# TorchToolkit

[![PyPI version fury.io](https://badge.fury.io/py/torchtoolkit.svg)](https://pypi.python.org/pypi/torchtoolkit/)

Hi 👋, this is a small Python package which contains useful function to use with PyTorch.
It includes [utilities](torchtoolkit/utils.py), [metrics](torchtoolkit/metrics.py) and [sampling](torchtoolkit/sampling.py) methods to use during and after training a model.

Feel free to use it, take the code for your projects, and raise an issue if you have question or a pull request if you want to contribute.

```shell
pip install torchtoolkit
```

Simplest example:

```python
from torchtoolkit import Transformer
from torch import randint

model = Transformer(num_encoder_layers=8, num_decoder_layers=8, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.2,
                    num_classes=5000, padding_token=0)

x = randint(low=0, high=5000, size=(400, 16))
y = model(x, auto_padding_mask=True)
```

I built it for my own usage, so you won't find documentation besides the docstring.
