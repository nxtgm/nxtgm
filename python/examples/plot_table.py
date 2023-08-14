"""
Table
===========================

This example finds and optimized seating arrangement for a dinner party.
The variables are the chairs and the labels are the persons.
The unary factors encode how much each person likes each chair.
The binary factors encode how much each person likes the person next to them.
The contraints ensure that a person can only sit on one chair
"""
from __future__ import annotations

import numpy as np
import nxtgm
# this example assume there are less or qual number of seats than persons
n_persons = 15
n_seats = 10
assert n_seats <= n_persons

np.random.seed(0)
# %%
# each person can prefer a table position
# we will encode this as a n_persons x n_seats matrix
person_seat_preference = np.random.rand(n_persons, n_seats) * 2 - 1

# %%
# n_persons x n_persons matrix to encode how much each person
# likes each other person. The eneries are in [-1 , 1] where
# -1 means they hate each other and 1 means they love each other.
# 0 means they are indifferent
person_person_preference = np.random.rand(n_persons, n_persons) * 2 - 1
person_person_preference = (
    person_person_preference + person_person_preference.T
) / 2


# %%
# create a graphical model with n_seats variables
# each variable can have n_persons labels
gm = nxtgm.DiscreteGm(num_var=n_seats, num_labels=n_persons)

# %%
# add a unary factor for each seat
# the unary factor encodes how much each person likes each seat
for seat in range(n_seats):

    values = person_seat_preference[:, seat]
    assert values.shape == (n_persons,)
    function_id = gm.add_function(values)
    gm.add_factor([seat], function_id)

# %%
# Add a binary factor for neighboring seats.
# The binary factor encodes how much each person likes the person next to them
# we assume a round tablel.
# The value table for the binary factor is a n_persons x n_persons matrix
# and is the same for all binary factors

function_id = gm.add_function(person_person_preference)

for seat in range(n_seats):

    left_seat = seat - 1
    if left_seat < 0:
        left_seat = n_seats - 1

    right_seat = seat + 1
    if right_seat >= n_seats:
        right_seat = 0

    # left seat
    variables = [seat, left_seat]
    gm.add_factor(variables, function_id)

    # right seat
    variables = [seat, right_seat]
    gm.add_factor(variables, function_id)

# %%
# constraints so that each person is only seated once
# so we need a constraint for each pair of seats
constraint_function = nxtgm.UniqueLables(
    arity=gm.num_variables, num_labels=n_persons,
)
constrain_function_id = gm.add_constraint_function(constraint_function)
variables = list(range(gm.num_variables))
gm.add_constraint(variables, constrain_function_id)


# optimize the model with Matching-ICM
Optimizer = nxtgm.MatchingIcm
parameters = dict(subgraph_size=2)
optimizer = Optimizer(gm, parameters)
callack = Optimizer.ReporterCallback(optimizer)
optimizer.optimize(callack)
best_solution = optimizer.best_solution()
print(best_solution)


# optimize the model with Matching-ICM
Optimizer = nxtgm.MatchingIcm
parameters = dict(subgraph_size=3)
optimizer = Optimizer(gm, parameters)
callack = Optimizer.ReporterCallback(optimizer)
optimizer.optimize(callack)
best_solution = optimizer.best_solution()
print(best_solution)


# %%
# optimize with an ILP solver
Optimizer = nxtgm.IlpHighs
optimizer = Optimizer(gm)
callack = Optimizer.ReporterCallback(optimizer)
optimizer.optimize(callack)
best_solution = optimizer.best_solution()
print(best_solution)
