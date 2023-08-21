#include <doctest/doctest.h>
#include <nxtgm/models/gm/discrete_gm/testing/optimizer_tester.hpp>

namespace nxtgm
{

TEST_CASE("chained_optimizers")
{

    OptimizerParameters icm_params;
    icm_params["time_limit"] = 10000000;

    OptimizerParameters belief_propagation_params;
    belief_propagation_params["max_iterations"] = 100;
    belief_propagation_params["convergence_tolerance"] = 0.0001;
    belief_propagation_params["damping"] = 0.5;
    belief_propagation_params["normalize_messages"] = true;

    OptimizerParameters chained_optimizer_params;
    chained_optimizer_params["time_limit_ms"] = 10000000;

    // order will be respected
    chained_optimizer_params["belief_propagation"] = belief_propagation_params;
    chained_optimizer_params["icm"] = icm_params;

    // clang-format off
    test_discrete_gm_optimizer(
        "chained_optimizers",
        chained_optimizer_params,
        potts_grid(4,4,2,false),
        require_local_optimality(true)
    );
    // clang-format on
}

TEST_CASE("belief_propagation")
{
    SUBCASE("trees")
    {
        // clang-format off
        test_discrete_gm_optimizer(
            "belief_propagation",
            OptimizerParameters(),
            {
                potts_grid(8, 1, 2, false),
                potts_grid(5, 1, 3, false),
            },
            {
                require_optimality(/*proven*/ false),
                require_optimization_status(OptimizationStatus::CONVERGED)
            }
        );
        // clang-format on
    }
    SUBCASE("trees_with_damping")
    {
        // clang-format off
        OptimizerParameters parameters;
        parameters["damping"] = 0.5;
        test_discrete_gm_optimizer(
            "belief_propagation",
            parameters,
            {
                potts_grid(8, 1, 2, false),
                potts_grid(5, 1, 3, false),
            },
            {
                require_optimality(/*proven*/ false),
                require_optimization_status(OptimizationStatus::CONVERGED)
            }
        );
        // clang-format on
    }
    SUBCASE("large_trees")
    {
        // clang-format off
        test_discrete_gm_optimizer(
            "belief_propagation",
            OptimizerParameters(),
            {
                potts_grid(30, 1, 2, false),
                potts_grid(20, 1, 3, false),
            },
            {
                require_optimization_status(OptimizationStatus::CONVERGED),
                require_local_n_optimality(3)
            }
        );
        // clang-format on
    }
}

TEST_CASE("dynamic_programming")
{
    SUBCASE("trees")
    {
        // clang-format off
        test_discrete_gm_optimizer(
            "dynamic_programming",
            OptimizerParameters(),
            {
                potts_grid(8, 1, 2, false),
                potts_grid(5, 1, 3, false),
                star(5, 3)
            },
            {
                require_optimality(/*proven*/ true),
            }
        );
        // clang-format on
    }

    SUBCASE("large_trees")
    {
        // clang-format off
        test_discrete_gm_optimizer(
            "dynamic_programming",
            OptimizerParameters(),
            {
                potts_grid(15, 1, 2, false),
                potts_grid(20, 1, 3, false),
                star(20, 10)
            },
            {
                require_optimization_status(OptimizationStatus::OPTIMAL),
                require_local_n_optimality(3)
            }
        );
        // clang-format on
    }
}

TEST_CASE("brute_force_naive")
{
    SUBCASE("basics")
    {
        // clang-format off
        test_discrete_gm_optimizer(
            "brute_force_naive",
            OptimizerParameters(),
            {
                potts_grid(5, 1, 3, false),
                random_model(6, 6, 3, 3),
                potts_chain_with_label_costs(5,5),
                unique_label_chain(3,4)
            },
            {
                require_optimization_status(OptimizationStatus::OPTIMAL),
                require_local_n_optimality(3)
            }
        );
        // clang-format on
    }
    SUBCASE("time_limited")
    {

        nxtgm::OptimizerParameters parameters;
        // 50 milliseconds as seconds
        parameters["time_limit_ms"] = 50;

        // clang-format off
        test_discrete_gm_optimizer(
            "brute_force_naive",
            parameters,
            potts_grid(10, 10, 3, false),
            require_optimization_status(OptimizationStatus::TIME_LIMIT_REACHED)
        );
        // clang-format on
    }
    SUBCASE("infeasible")
    {
        // clang-format off
        test_discrete_gm_optimizer(
            "brute_force_naive",
            OptimizerParameters(),
            infeasible_model(),
            require_infesibility(true)
        );
        // clang-format on
    }
}

TEST_CASE("graph_cut")
{
    SUBCASE("small_grids")
    {
        // clang-format off
        test_discrete_gm_optimizer(
            "graph_cut",
            OptimizerParameters(),
            {
                potts_grid(3,4,2,true),
                potts_grid(10,1,2,true)
            },
            require_optimality(/*proven*/ true)
        );
        // clang-format on
    }

    SUBCASE("large_grids")
    {
        // clang-format off
        test_discrete_gm_optimizer(
            "graph_cut",
            OptimizerParameters(),
            potts_grid(5,5,2,true),
            {
                require_optimization_status(OptimizationStatus::OPTIMAL),
                require_local_n_optimality(3)
            }
        );
        // clang-format on
    }
}

TEST_CASE("icm")
{

    // clang-format off
    test_discrete_gm_optimizer(
        "icm",
        OptimizerParameters(),
        {
            potts_grid(5,5,3,false),
            potts_grid(7,7,2,false),
            potts_chain_with_label_costs(5,5),
            unique_label_chain(3,4),
            unique_label_chain(5,5),
            infeasible_model(),
        },
        require_local_optimality(true)
    );
    // clang-format on
}

TEST_CASE("qpbo")
{
    TestDiscreteGmOptimizerOptions test_options;
    test_options.seed = 42;

    OptimizerParameters parameters;
    parameters["strong_persistencies"] = 0;
    parameters["probing"] = 1;

    SUBCASE("small")
    {
        // clang-format off
        test_discrete_gm_optimizer(
            "qpbo",
            parameters,
            {
                potts_grid(3,4,2,true),
                potts_grid(10,1,2,false),
                star(5,2),
                sparse_potts_chain(4, 2)
            },
            require_optimality(/*proven*/ true),
            test_options
        );
        // clang-format on
    }

    SUBCASE("large")
    {
        // clang-format off
        test_discrete_gm_optimizer(
            "qpbo",
            parameters,
            {
                potts_grid(10,10,2,true),
                star(20,2)
            },
            {
                require_optimization_status(OptimizationStatus::OPTIMAL),
                require_local_n_optimality(3)
            }
        );
        // clang-format on
    }
}

TEST_CASE("matching_icm")
{

    nxtgm::OptimizerParameters params;
    params["subgraph_size"] = 2;

    for (auto subgraph_size = 2; subgraph_size <= 3; ++subgraph_size)
    {
        params["subgraph_size"] = subgraph_size;
        // clang-format off
        test_discrete_gm_optimizer(
            "matching_icm",
            OptimizerParameters(),
            {
                unique_label_chain(8,8),
                unique_label_chain(10,12),
            },
            {
                require_local_n_optimality(subgraph_size),
                require_optimization_status(OptimizationStatus::LOCAL_OPTIMAL)
            }
        );
        // clang-format on
    }
}
#ifdef _WIN32
#define SKIP_WIN doctest::skip(true)
#else
#define SKIP_WIN doctest::skip(false)
#endif

TEST_CASE("ilp_highs" * SKIP_WIN)
{

    SUBCASE("small")
    {
        // clang-format off
        test_discrete_gm_optimizer(
            "ilp_highs",
            OptimizerParameters(),
            {
                potts_grid(3,4,2,true),
                potts_grid(3,4,2,false),
                potts_grid(2,3,2,false),
                star(5,2),
                sparse_potts_chain(4, 2),
                random_sparse_model(4, 3, 2, 4, 4, 0.5 ),
                random_sparse_model(10,10, 2, 4, 4, 0.2 ),
                sparse_potts_chain(5,5),
                potts_chain_with_label_costs(5,5),
                unique_label_chain(2,2, true),
                unique_label_chain(4,5, true),
                unique_label_chain(2,2, false),
                unique_label_chain(4,5, false)
            },
            require_optimality(/*proven*/ true, /*tolerance*/ 1e-3)
        );
        // clang-format on
    }
}

} // namespace nxtgm
