from ._nxtgm import *
from ._nxtgm import __version__


# 



def _extend():
    
    # make parameter factory  class
    # which only takes keyword arguments
    def parameter_factory(self, **kwargs):
        parmeters  = self.Parameters()
        for k,v in kwargs.items():
            setattr(parmeters, k, v)
        return parmeters


    all_solvers = [
        IlpHighs,
        DynamicProgramming,
        BruteForceNaive
    ]
    for solver in all_solvers:
        cls_name = solver.__name__

        # add the parameters class to the solver
        solver.Parameters = getattr(_nxtgm, cls_name + "Parameters") 

        # report callback
        solver.ReporterCallback = DiscreteGmOptimizerReporterCallback

        solver.parameters = classmethod(parameter_factory)

    pass
_extend()
del _extend 