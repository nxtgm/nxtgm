import nxtgm
import numpy as np


def add_random_unaries(gm, seed=0):
    np.random.seed(seed)
    num_variables = len(gm.space)

    for i in range(num_variables):
        unaries = np.random.rand(gm.space[i])
        function_id = gm.add_function(unaries)
        gm.add_factor([i], function_id)


def potts_chain(num_variables, num_labels, seed=0):
    
    np.random.seed(seed)
    gm = nxtgm.DiscreteGm(num_variables, num_labels)

    # unaries 
    add_random_unaries(gm, seed=seed)

    # potts
    for i in range(num_variables-1):
        # random beta 
        beta = np.random.rand()
        potts = nxtgm.Potts(num_labels, beta)
        function_id = gm.add_function(potts)
        gm.add_factor([i,i+1], function_id)
    
    return gm


def potts_grid(shape, num_labels, seed=0):
    
    def get_vis(x,y):
        return x*shape[1] + y

    np.random.seed(seed)
    num_variables = np.prod(shape)
    gm = nxtgm.DiscreteGm(num_variables, num_labels)

    # unaries 
    add_random_unaries(gm, seed=seed)

    # potts
    for x in range(shape[0]):
        for y in range(shape[1]):

            # right
            if x < shape[0]-1:
                beta = np.random.rand()
                potts = nxtgm.Potts(num_labels, beta)
                function_id = gm.add_function(potts)
                vis = [get_vis(x,y), get_vis(x+1,y)]
                gm.add_factor(vis, function_id)

            # down
            if y < shape[1]-1:
                beta = np.random.rand()
                potts = nxtgm.Potts(num_labels, beta)
                function_id = gm.add_function(potts)
                vis = [get_vis(x,y), get_vis(x,y+1)]
                gm.add_factor(vis, function_id)
                