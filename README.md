# Flexformer

A general implementation of Transformer, to play around with attention and build custom architectures

It is built to be versatile, allowing to use several attention mechanisms, positional encoding and encoder / decoder layers.

I built this package for my own experiments, so you won't find documentation (expect docstrings), but don't hesitate to open an issue if you have any question or want to report a bug.

Flexformer implements :

* Linear attention, with Elu and Favor+ feature maps
* Rotary positional encoding (used by default)
* Relative positional encoding
