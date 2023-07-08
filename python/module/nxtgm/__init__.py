from __future__ import annotations

from ._nxtgm import *  # noqa: F401, F403
from ._nxtgm import __version__  # noqa: F401


#


def _extend():

    # make parameter factory  class
    # which only takes keyword arguments
    def parameter_factory(self, **kwargs):
        parmeters = self.Parameters()
        for k, v in kwargs.items():
            setattr(parmeters, k, v)
        return parmeters

    all_solvers = [
        IlpHighs,  # noqa: F405
        DynamicProgramming,  # noqa: F405
        BruteForceNaive,  # noqa: F405
        Icm,  # noqa: F405
    ]
    for solver in all_solvers:
        cls_name = solver.__name__

        # add the parameters class to the solver
        solver.Parameters = \
            getattr(_nxtgm, cls_name + 'Parameters')  # noqa: F405

        # report callback
        solver.ReporterCallback = \
            DiscreteGmOptimizerReporterCallback  # noqa: F405

        solver.parameters = classmethod(parameter_factory)

    pass


_extend()
del _extend
