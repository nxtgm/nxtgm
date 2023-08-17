from __future__ import annotations

import numbers

from ._nxtgm import *  # noqa: F401, F403
from ._nxtgm import __version__  # noqa: F401
from ._nxtgm import _discrete_gm_optimizer_factory
from ._nxtgm import _OptimizerParameters


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
