from __future__ import annotations

import numbers
import os
import sys

from ._nxtgm import *  # noqa: F401, F403
from ._nxtgm import __version__  # noqa: F401
from ._nxtgm import _discrete_gm_optimizer_factory
from ._nxtgm import _OptimizerParameters
from ._nxtgm import _proposal_gen_factory


# on emscripten, the PREFIX is **always** the root of the filesystem
# ie "/".  This is used to find the plugins
if sys.platform == 'emscripten':
    os.environ['CONDA_PREFIX'] = '/'


class OptimizerParameters(_OptimizerParameters):
    # init with **only** keyword arguments
    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            # integer like via numbers
            if isinstance(value, numbers.Integral):
                self[key] = int(value)
            elif isinstance(value, numbers.Real):
                self[key] = float(value)
            elif isinstance(value, str):
                self[key] = value
            # dict like
            elif isinstance(value, dict):
                param = OptimizerParameters(**value)
                self[key] = param
            else:
                raise TypeError(
                    f'Unsupported type {type(value)} for parameter {key}',
                )


def discrete_gm_optimizer_factory(
    gm, optimizer_name,
    parameters=OptimizerParameters(),
):
    if isinstance(parameters, dict):
        parameters = OptimizerParameters(**parameters)

    return _discrete_gm_optimizer_factory(
        gm=gm, optimizer_name=optimizer_name,
        parameters=parameters,
    )


class _ProposalGeneratorFactory:
    def __init__(self, cls, **kwargs):
        self._cls = cls
        self._kwargs = kwargs
        self.license = 'MIT'
        self.description = 'pure python proposal generator factory'

    def create(self, gm):
        return self._cls(gm=gm, **self._kwargs)


# proposal generator related
def proposal_gen_factory(proposal_gen_cls, *args, **kwargs):
    _factory = _ProposalGeneratorFactory(proposal_gen_cls, *args, **kwargs)
    return _proposal_gen_factory(_factory)
