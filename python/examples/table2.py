from __future__ import annotations

import sys

import numpy
import nxtgm
sys.path.append('/Users/thorstenbeier/src/nxtgm/python/module/')


def make_room_graph(n_seats):
    table_graph = numpy.zeros((n_seats, n_seats))
    for i in range(n_seats):
        table_graph[i, (i + 1) % n_seats] = 1
        table_graph[(i + 1) % n_seats, i] = 1
    return table_graph


if __name__ == '__main__':
    numpy.random.seed(42)
    # nice numpy print
    numpy.set_printoptions(precision=2, suppress=True)

    # random dataset with n_persons and n_seats
    n_seats = 300
    n_persons = 100
    # names = [random_name() for _ in range(n_persons)]

    # groups:
    # - we have n_groups groups
    # - each person can be in a group or not
    # - each person can only be in multiple groups
    # - considers groups smth like a family, a work group, a sports group, etc.
    n_groups = 10
    person_in_group = numpy.random.randint(2, size=(n_persons, n_groups))

    # each person can like or dislike each other person
    # this is encoded as a n_persons x n_persons matrix
    # a negative value means dislike, a positive value means like
    # 0 means indifferent
    # (the label 0 is reserved for "no person",
    # but one can still encode preferences for this label)
    # sparsity = 0.8
    person_person_preference = numpy.random.rand(
        n_persons+1, n_persons+1,
    ) * 2 - 1

    person_seat_preference = numpy.random.rand(n_persons, n_seats) * 2 - 1
    # print(person_seat_preference)

    # the geometry of the tables is encoded as graph
    # each node represents a seat
    # each edge represents a connection between two seats
    # the edges are undirected and have weights
    # the weights are how relevant this edge is
    # ie closer seats have higher weights

    # room graph
    room_graph = make_room_graph(n_seats)

    # create a graphical model with n_seats variables
    # each variable can have n_persons + 1 labels
    # the first label is "no person"
    gm = nxtgm.DiscreteGm(num_var=n_seats, num_labels=n_persons + 1)

    # add a unary factor for each seat
    # the unary factor encodes how much each person likes each seat
    for seat in range(n_seats):
        # the unary factor is a n_persons+1 vector
        # the first entry is the cost of no person sitting here
        # the other entries are the costs of each person sitting here
        values = person_seat_preference[:, seat]
        assert values.shape == (n_persons,)

        costs = numpy.zeros((n_persons+1,))
        costs[1:] = -values

        function_id = gm.add_function(costs)
        gm.add_factor([seat], function_id)

    # pairwise costs encode how much each person
    # likes to sit next to each other person
    # "next" is defined by the room graph
    for seat_index_u in range(n_seats):
        for seat_index_v in range(n_seats):
            if room_graph[seat_index_u, seat_index_v] > 0:
                weight = room_graph[seat_index_u, seat_index_v]
                variables = [seat_index_u, seat_index_v]

                # prefereces to costs
                costs = -person_person_preference * weight
                function_id = gm.add_function(costs)
                gm.add_factor(variables, function_id)

    # add constraint that each person can only sit on one seat
    # we need a constraint for each pair of seats or a global constraint
    # for all seats, here we use a global constraint
    vars = list(range(n_seats))
    constraint_function = nxtgm.UniqueLables(
        arity=gm.num_variables,
        num_labels=n_persons+1,
        with_ignore_label=True,
        ignore_label=0,
    )
    gm.add_constraint(vars, gm.add_constraint_function(constraint_function))

    # print that gm
    print(f"{gm.num_variables=} {gm.num_factors=} {gm.num_constraints=}")

    # proposal generator
    proposal_gen_name = 'random'
    proposal_gen_parameters = nxtgm.OptimizerParameters()
    proposal_gen_parameters['num_iterations'] = 100000
    proposal_gen_parameters['exit_after_n_rejections'] = 1000
    proposal_gen_parameters['p_flip'] = 0.5
    proposal_gen_parameters['seed'] = 42

    # fusion move optimizer
    fusion_move_optimizer_name = 'qpbo'
    fusion_move_optimizer_parameters = nxtgm.OptimizerParameters()

    # fusion move parameters
    fusion_parameters = nxtgm.OptimizerParameters()
    fusion_parameters['num_iterations'] = 100000
    # noqa: E501
    fusion_parameters['optimizer_parameters'] = \
        fusion_move_optimizer_parameters
    fusion_parameters['optimizer_name'] = fusion_move_optimizer_name
    fusion_parameters['numeric_stability'] = True

    optimizer_name = 'fusion_moves'
    fusion_move_parameters = nxtgm.OptimizerParameters()
    fusion_move_parameters['proposal_gen_name'] = proposal_gen_name
    fusion_move_parameters['proposal_gen_parameters'] = proposal_gen_parameters
    fusion_move_parameters['fusion_parameters'] = fusion_parameters

    optimizer_parameters = fusion_move_parameters
    optimizer = nxtgm.discrete_gm_optimizer_factory(
        gm, optimizer_name, optimizer_parameters,
    )

    # starting point (random)
    starting_point = numpy.random.randint(1, n_persons+1, size=(n_seats,))

    # starting_point[:]=0
    reporter_callback = nxtgm.DiscreteGmOptimizerReporterCallback(optimizer)
    optimizer.optimize(
        reporter_callback=reporter_callback,
        starting_point=starting_point,
    )
