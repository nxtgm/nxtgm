from . import conftest
from . testmodels import *

import pytest
import numpy as np
import datetime
import nxtgm


def test_nxtgm_version():
    v = nxtgm.__version__
    parts = v.split('.')
    assert len(parts) == 3
    assert parts[0].isdigit()
    assert parts[1].isdigit()
    assert parts[2].isdigit()


class TestDiscreteSpace(object):

    def test_simple_space(self):
        gm = nxtgm.DiscreteGm(3,2)
        space = gm.space
        assert len(space) == 3
        assert space.is_simple
        assert space[0] == 2
        assert space[1] == 2
        assert space[2] == 2
    
    def test_non_simple_space(self):
        n_labels = [2, 3, 4, 5]
        gm = nxtgm.DiscreteGm(n_labels)
        space = gm.space
        assert len(space) == 4
        assert not space.is_simple
        assert space[0] == 2
        assert space[1] == 3
        assert space[2] == 4
        assert space[3] == 5



class TestPotts(object):

    def test_basics(self):
        f = nxtgm.Potts(num_labels=3, beta=0.5)
        assert f.arity == 2
        assert f.size == 9
        assert f.shape == (3,3)
        assert f[0,0] == pytest.approx(0.0)
        assert f[0,1] == pytest.approx(0.5)
        assert f[0,2] == pytest.approx(0.5)
        assert f[1,0] == pytest.approx(0.5)
        assert f[1,1] == pytest.approx(0.0)
        assert f[1,2] == pytest.approx(0.5)
        assert f[2,0] == pytest.approx(0.5)
        assert f[2,1] == pytest.approx(0.5)
        assert f[2,2] == pytest.approx(0.0)


class TestLabelCosts(object):
    
        def test_basics(self):
            arity = 5
            num_labels = 10
            label_costs = np.arange(num_labels) + 1
            f = nxtgm.LabelCosts(arity=arity, label_costs=label_costs)
            assert f.arity == arity
            assert f.size == 10**5
            assert f.shape == tuple([10] * 5)
            assert f[0,0,0,0,0] == pytest.approx(1)
            assert f[0,0,0,0,1] == pytest.approx(1+2)
            assert f[0,0,0,0,2] == pytest.approx(1+3)
            assert f[0,0,3,0,3] == pytest.approx(1+4)
            assert f[0,1,2,3,4] == pytest.approx(1+2+3+4+5)
            assert f[4,4,4,4,4] == pytest.approx(5)
        

class TestDiscreteGm(object):

    def test_max_properties(self):
    
        gm = nxtgm.DiscreteGm(10, 2)
        assert gm.max_factor_arity == 0
        assert gm.max_factor_size == 0
        assert gm.max_constraint_arity == 0
        assert gm.max_constraint_size == 0
    

    def test_add_function(self):
        gm = nxtgm.DiscreteGm(10, 3)
        function_id = gm.add_function(np.array([1,2,3]))
        assert function_id == 0
        factor_id = gm.add_factor([0], function_id)
        assert factor_id == 0
        assert gm.max_factor_arity == 1
        assert gm.max_factor_size == 3

    
    @pytest.mark.parametrize("array", [np.ones([3]), np.ones([3,3]),np.ones([3,3,3])])
    def test_add_function_from_numpy(self, array):
        gm = nxtgm.DiscreteGm(10, 3)
        function_id = gm.add_function(array)
        assert function_id == 0
        vis = np.arange(array.ndim)
        factor_id = gm.add_factor(vis, function_id)
        assert factor_id == 0
        assert gm.max_factor_arity == array.ndim
        assert gm.max_factor_size == array.size

    
    def test_evaluate_simple(self):
        
        gm = nxtgm.DiscreteGm(2, 2)
        potts = nxtgm.Potts(num_labels=2, beta=1.0)
        function_id = gm.add_function(potts)
        gm.add_factor([0,1], function_id)

        assert len(gm.evaluate([0,0])) == 2

        
        # enery
        assert gm.evaluate([0,0])[0] == pytest.approx(0.0)
        assert gm.evaluate([0,1])[0] == pytest.approx(1.0)

        # how violated
        assert gm.evaluate([0,0])[1] == pytest.approx(0.0)
        assert gm.evaluate([0,1])[1] == pytest.approx(0.0)




class TestOptimizers(object):

    def test_ilp_highs(self):
        gm = potts_chain(num_variables=10, num_labels=2)


        # setup optimizer
        solver_cls = nxtgm.IlpHighs
        optimizer = solver_cls(gm, solver_cls.parameters(integer=True, time_limit=datetime.timedelta(seconds=10.0)))

        # setup reporter callback
        reporter_callback = solver_cls.ReporterCallback(optimizer)
        
        #let the optimizer do its work
        print("optimizing:");
        status = optimizer.optimize(reporter_callback)
        assert  status == nxtgm.OptimizationStatus.OPTIMAL

        # get solution  
        solution = optimizer.best_solution()
        assert solution is not None
        print("solution: ", solution)
        assert solution.shape[0] == len(gm.space)
        energy, how_violated = gm.evaluate(solution)
        print(f"energy: {energy}, how_violated: {how_violated}")