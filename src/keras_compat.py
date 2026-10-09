"""Compatibility shim for loading the PneumoScan checkpoint.

models/best_model.h5 was saved with Keras 3.14.1, which writes
legacy no-op keys into H5 layer configs (e.g. 'renorm',
'renorm_clipping', 'renorm_momentum', 'synchronized' on
BatchNormalization and 'quantization_config' on Dense). The
Keras versions currently installable from PyPI (e.g. 3.12.x)
accept **kwargs in Layer.__init__ but raise on unknown keys,
so load_model() fails on the checkpoint.

This module patches Layer.__init__ to keep only the keyword
arguments it declares, dropping those legacy no-op keys. The
keys carry no learned state — all weights (kernels, biases,
gamma/beta, moving statistics) are stored in the H5 weight
groups and load by layer name, so inference behaviour is
unchanged.

Import this module before calling load_model():

    import keras_compat  # noqa: F401
"""

import inspect

from keras.layers import Layer

_orig_layer_init = Layer.__init__


def _layer_init_compat(self, *args, **kwargs):
    params = inspect.signature(_orig_layer_init).parameters
    allowed = {
        name for name, p in params.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      inspect.Parameter.KEYWORD_ONLY)
    }
    kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    _orig_layer_init(self, *args, **kwargs)


Layer.__init__ = _layer_init_compat
