from __future__ import annotations

from ._nxtgm import *  # noqa: F401, F403
from ._nxtgm import __version__  # noqa: F401


#


def _extend():

    all_solvers = [
        IlpHighs,  # noqa: F405
        DynamicProgramming,  # noqa: F405
        BruteForceNaive,  # noqa: F405
        Icm,  # noqa: F405
        MatchingIcm,  # noqa: F405
    ]
    for solver in all_solvers:
        # report callback
        solver.ReporterCallback = \
            DiscreteGmOptimizerReporterCallback  # noqa: F405

    pass


_extend()
del _extend
