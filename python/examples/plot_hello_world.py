"""
Hello World
===========================

This example introduces the basic usage of nxtgm.
"""
from __future__ import annotations

import numpy as np
import nxtgm


# %%
# The absolute minimal (pointless) example
# ------------------------
#
# Here we create a graphical model with a single variable
# which can have two values, 0 or 1.

gm = nxtgm.DiscreteGm(num_var=1, num_labels=2)

# %%
# We can now  add a single unary factor to the model.
# The factor is a simple table with two entries.
# The first entry is the energy for the variable being 0,
# the second entry is the energy for the variable being 1.
# the "value_table" for a factor is called "function".
# we first add such a function
function_id = gm.add_function(np.array([42.0, 30.0]))

# %%
# We can now add a factor to the model.
variables = [0]  # the factor is connected to the only variable in the model
gm.add_factor(variables, function_id)


# %%
# We can now optimize the model and find the best labels for all
# variables in the model. In this case its a single variable.
# The result is a numpy array with the best label for each variable.
# Since this model is very simple (and the result is obvious) we
# use a brute force optimizer.

optimizer = nxtgm.BruteForceNaive

optimizer = nxtgm.BruteForceNaive(gm)
optimizer.optimize()


# %%
# Get the best solution from the optimizer
# in our case it will be [1] since the energy for the variable being 0 is  42.0
# and the energy for the variable being 1 is 30.0
best_solution = optimizer.best_solution()
best_solution
