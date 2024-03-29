#include <doctest/doctest.h>
#include <nxtgm/models/gm/discrete_gm/testing/optimizer_tester.hpp>
#include <nxtgm/plugins/ilp/ilp_base.hpp>
#include <nxtgm/plugins/plugin_registry.hpp>
#include <nxtgm_test_common.hpp>
#ifdef _WIN32
#define SKIP_WIN doctest::skip(true)
#else
#define SKIP_WIN doctest::skip(false)
#endif

namespace nxtgm
{

// TEST_CASE("raise_on_unknown_parameters" * doctest::skip(true))
// {
//     for (auto optimizer_name : all_optimizers())
//     {
//         OptimizerParameters parameters;
//         parameters["_unknown_parameter"] = 42;

//         auto model_and_name = potts_grid(1, 2, 2, true)->operator()(0 /*seed*/);
//         auto model = std::move(model_and_name.first);

//         INFO(optimizer_name);
//         // CHECK_THROWS_AS(discrete_gm_optimizer_factory(model, optimizer_name, parameters),
//         UnknownParameterException);

//         bool did_throw = false;
//         bool wrong_exception = false;
//         try
//         {
//             auto optimizer = discrete_gm_optimizer_factory(model, optimizer_name, parameters);
//         }
//         catch (const UnknownParameterException &e)
//         {
//             did_throw = true;
//         }
//         catch (const std::exception &e)
//         {
//             // INFO(optimizer_name, "wrong exception", std::string(e.what()));
//             wrong_exception = true;
//         }
//         CHECK(wrong_exception == false);
//         CHECK(did_throw);
//     };
// }

// TEST_CASE("raise_on_unsupported_model" * doctest::skip(true))
// {
//     for (auto optimizer_name : all_optimizers())
//     {
//         OptimizerParameters parameters;

//         auto model_and_name = unique_label_chain(10, 3)->operator()(0 /*seed*/);
//         auto model = std::move(model_and_name.first);

//         try
//         {
//             auto optimizer = discrete_gm_optimizer_factory(model, optimizer_name, parameters);
//         }
//         catch (const UnsupportedModelException &e)
//         {
//         }
//         catch (const std::exception &e)
//         {
//             INFO(optimizer_name, "wrong exception", std::string(e.what()));
//             CHECK(false);
//         }
//     }
// }

TEST_CASE("chained_optimizers"){

    SUBCASE("basic"){OptimizerParameters icm_params;
icm_params["time_limit_ms"] = 10000000;

OptimizerParameters belief_propagation_params;
belief_propagation_params["max_iterations"] = 100;
belief_propagation_params["convergence_tolerance"] = 0.0001;
belief_propagation_params["damping"] = 0.5;
belief_propagation_params["normalize_messages"] = true;

OptimizerParameters chained_optimizer_params;
chained_optimizer_params["time_limit_ms"] = 10000000;

// order will be respected
chained_optimizer_params["optimizer_names"].push_back("belief_propagation");
chained_optimizer_params["optimizer_names"].push_back("icm");

chained_optimizer_params["optimizer_parameters"].push_back(belief_propagation_params);
chained_optimizer_params["optimizer_parameters"].push_back(icm_params);

// clang-format off
            test_discrete_gm_optimizer(
                "chained_optimizers",
                chained_optimizer_params,
                potts_grid(4,4,2,false),
                require_local_optimality(true)
            );
// clang-format on
} // namespace nxtgm
#ifndef WIN32
SUBCASE("icm_ilp")
{
    for (auto ilp_plugin : all_ilp_plugins())
    {
        SUBCASE(ilp_plugin.c_str())
        {
            OptimizerParameters ilp_based_parameters;
            ilp_based_parameters["ilp_plugin_name"] = ilp_plugin;

            OptimizerParameters chained_optimizer_params;
            chained_optimizer_params["optimizer_names"].push_back("icm");
            chained_optimizer_params["optimizer_parameters"].push_back(OptimizerParameters());

            chained_optimizer_params["optimizer_names"].push_back("ilp_based");
            chained_optimizer_params["optimizer_parameters"].push_back(ilp_based_parameters);

            // clang-format off
                test_discrete_gm_optimizer(
                    "chained_optimizers",
                    chained_optimizer_params,
                    potts_grid(4,4,2,false),
                    {
                        require_optimality(),
                        require_optimization_status(OptimizationStatus::OPTIMAL)
                    }
                );
            // clang-format on
        }
    }
}
#endif
}

TEST_CASE("belief_propagation")
{
    SUBCASE("fancy_models")
    {

        OptimizerParameters belief_propagation_params;
        belief_propagation_params["max_iterations"] = 3;
        belief_propagation_params["convergence_tolerance"] = 0.0001;
        belief_propagation_params["damping"] = 0.999;
        belief_propagation_params["normalize_messages"] = true;

        test_discrete_gm_optimizer("belief_propagation", belief_propagation_params,
                                   {unique_label_chain(5, 5, false /*pairwise*/, true /*with ignore label*/)},
                                   {
                                       // require_optimization_status(OptimizationStatus::CONVERGED)
                                   });
        std::cout << "done fancy" << std::endl;
    }

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
                require_optimality(),
                require_optimization_status(OptimizationStatus::CONVERGED)
            }
        );
        // clang-format on
    }
    SUBCASE("large_chains")
    {
        // clang-format off
        test_discrete_gm_optimizer(
            "belief_propagation",
            OptimizerParameters(),
            {
                potts_grid(20, 1, 10, false),
                potts_grid(20, 1, 10, false),
            },
            {
                // compare against dynamic programming
                require_optimality( /*tolerance*/ 1e-3, "dynamic_programming"),
                require_optimization_status(OptimizationStatus::CONVERGED)
            }
        );
        // clang-format on
    }
    SUBCASE("forest")
    {
        // clang-format off
        test_discrete_gm_optimizer(
            "belief_propagation",
            OptimizerParameters(),
            {
                concatenated_models(
                    std::move(potts_grid(8, 1, 2, false)),
                    std::move(potts_grid(5, 1, 2, false))
                ),
                concatenated_models(
                    std::move(potts_grid(3, 1, 2, false)),
                    std::move(potts_grid(3, 1, 3, false)),
                    std::move(star(5, 2))
                ),
            },
            {
                require_optimality(),
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
                require_optimality(),
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
                require_optimality(),
                require_optimization_status(OptimizationStatus::OPTIMAL)
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
                star(8, 5)
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
            {
                require_optimality( /*tolerance*/ 1e-3),
                require_optimization_status(OptimizationStatus::OPTIMAL)
            }
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

    SUBCASE("small")
    {
        OptimizerParameters parameters;
        parameters["strong_persistencies"] = 0;
        // parameters["probing"] = 1;

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
            {
                require_optimality( /*tolerance*/ 1e-3),
                require_optimization_status(OptimizationStatus::OPTIMAL)
            }
        );
        // clang-format on
    }
    SUBCASE("large")
    {
        OptimizerParameters parameters;
        parameters["strong_persistencies"] = 0;

        // clang-format off
        test_discrete_gm_optimizer(
            "qpbo",
            parameters,
            {
                potts_grid(6,6,2,true),
                star(10,2)
            },
            {
                require_optimization_status(OptimizationStatus::OPTIMAL),
                require_local_n_optimality(3)
            }
        );
        // clang-format on
    }
    SUBCASE("partial_optimality_black_box")
    {
        OptimizerParameters parameters;
        parameters["strong_persistencies"] = 0;

        // clang-format off
        test_discrete_gm_optimizer(
            "qpbo",
            parameters,
            {
                potts_grid(3,4,2,false),
                potts_grid(12,1,2,false),
                star(11,2),
                sparse_potts_chain(12, 2)
            },
            require_correct_partial_optimality()
        );
        // clang-format on
    }
    SUBCASE("partial_optimality")
    {
        OptimizerParameters parameters;
        parameters["strong_persistencies"] = 0;

        for (unsigned seed = 0; seed < 42; ++seed)
        {
            auto [easy_gm, easy_gm_name] = potts_grid(100, 1, 2, true)->operator()(seed);
            auto [hard_gm, hard_gm_name] = potts_grid(100, 100, 2, false)->operator()(seed);
            std::vector<DiscreteGm> models;
            models.push_back(std::move(easy_gm));
            models.push_back(std::move(hard_gm));

            auto concated_gm = concat_models(models);

            auto expected_optimizer = discrete_gm_optimizer_factory(concated_gm, "qpbo", parameters);
            if (!expected_optimizer)
            {
                throw std::runtime_error(expected_optimizer.error());
            }
            auto optimizer = std::move(expected_optimizer.value());
            auto status = optimizer->optimize();

            for (auto i = 0; i < models[0].num_variables(); ++i)
            {
                CHECK(optimizer->is_partial_optimal(i));
            }
        }
    }
}

TEST_CASE("hqpbo")
{
    TestDiscreteGmOptimizerOptions test_options;
    test_options.seed = 42;

    SUBCASE("second_order")
    {
        SUBCASE("small")
        {
            OptimizerParameters parameters;
            parameters["strong_persistencies"] = 0;
            // parameters["probing"] = 1;

            // clang-format off
            test_discrete_gm_optimizer(
                "hqpbo",
                parameters,
                {
                    //random_sparse_model(3, 1, 3, 3, 2, 0.9)//,
                    potts_grid(4,4,2,true),
                    potts_grid(2,1,2,false),
                    star(5,2),
                    sparse_potts_chain(4, 2)
                },
                {
                    require_optimality( /*tolerance*/ 1e-3),
                    require_optimization_status(OptimizationStatus::OPTIMAL)
                }
            );
            // clang-format on
        }
        SUBCASE("large")
        {
            OptimizerParameters parameters;
            parameters["strong_persistencies"] = 0;

            // clang-format off
            test_discrete_gm_optimizer(
                "hqpbo",
                parameters,
                {
                    potts_grid(20,20,2,true),
                    star(20,2)
                },
                {
                    require_optimization_status(OptimizationStatus::OPTIMAL),
                    require_local_n_optimality(1)
                }
            );
            // clang-format on
        }
        SUBCASE("partial_optimality_black_box")
        {
            OptimizerParameters parameters;
            parameters["strong_persistencies"] = 0;

            // clang-format off
            test_discrete_gm_optimizer(
                "hqpbo",
                parameters,
                {
                    potts_grid(3,4,2,false),
                    potts_grid(12,1,2,false),
                    star(11,2),
                    sparse_potts_chain(12, 2)
                },
                require_correct_partial_optimality()
            );
            // clang-format on
        }
        SUBCASE("partial_optimality")
        {
            OptimizerParameters parameters;
            parameters["strong_persistencies"] = 0;

            for (unsigned seed = 0; seed < 42; ++seed)
            {
                auto [easy_gm, easy_gm_name] = potts_grid(100, 1, 2, true)->operator()(seed);
                auto [hard_gm, hard_gm_name] = potts_grid(100, 100, 2, false)->operator()(seed);
                std::vector<DiscreteGm> models;
                models.push_back(std::move(easy_gm));
                models.push_back(std::move(hard_gm));

                auto concated_gm = concat_models(models);

                auto expected_optimizer = discrete_gm_optimizer_factory(concated_gm, "qpbo", parameters);
                if (!expected_optimizer)
                {
                    throw std::runtime_error(expected_optimizer.error());
                }
                auto optimizer = std::move(expected_optimizer.value());
                auto status = optimizer->optimize();

                for (auto i = 0; i < models[0].num_variables(); ++i)
                {
                    CHECK(optimizer->is_partial_optimal(i));
                }
            }
        }
    }
    SUBCASE("higher_order")
    {
        SUBCASE("partial_optimality_black_box")
        {
            OptimizerParameters parameters;
            parameters["strong_persistencies"] = 0;

            // clang-format off
            test_discrete_gm_optimizer(
                "hqpbo",
                parameters,
                {
                    random_model(12, 5, 3,  2),
                    random_model(12, 5, 10, 2)
                },
                require_correct_partial_optimality()
            );
            // clang-format on
        }
    }
}

TEST_CASE("ilp_based" * SKIP_WIN)
{
    for (auto ilp_plugin : all_ilp_plugins())
    {
        SUBCASE(ilp_plugin.c_str())
        {
            std::cout << "test ilp_based with " << ilp_plugin << std::endl;
            // clang-format off
            OptimizerParameters parameters;
            parameters["ilp_plugin_name"] = ilp_plugin;

            OptimizerParameters ilp_parameters;
            ilp_parameters["log_level"] = 0;
            parameters["ilp_plugin_parameters"] = ilp_parameters;
            test_discrete_gm_optimizer(
                "ilp_based",
                parameters,
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
                {
                    require_optimality( /*tolerance*/ 1e-3),
                    require_optimization_status(OptimizationStatus::OPTIMAL)
                }
            );
            // clang-format on
        }
    }
}

TEST_CASE("reduced_gm_optimizer")
{
    SUBCASE("second_order")
    {
        SUBCASE("brute_force_naive")
        {
            OptimizerParameters parameters;
            parameters["sub_optimizer"] = "brute_force_naive";

            // clang-format off
            test_discrete_gm_optimizer(
                "reduced_gm_optimizer",
                parameters,
                {
                    potts_grid(4,3,2,false)
                },
                {
                    require_optimality( /*tolerance*/ 1e-3),
                    require_optimization_status(OptimizationStatus::OPTIMAL)
                }
            );
            // clang-format on
        }
        SUBCASE("icm")
        {
            OptimizerParameters parameters;
            parameters["sub_optimizer"] = "icm";

            // clang-format off
            test_discrete_gm_optimizer(
                "reduced_gm_optimizer",
                parameters,
                {
                    potts_grid(10,10,2,false)
                },
                // return status needs to be local_optimal OR optimal
                require_local_optimality(/*proven*/true)
            );
            // clang-format on
        }
    }
    SUBCASE("higher_order")
    {
        SUBCASE("brute_force_naive")
        {
            OptimizerParameters parameters;
            parameters["sub_optimizer"] = "brute_force_naive";

            // clang-format off
            test_discrete_gm_optimizer(
                "reduced_gm_optimizer",
                parameters,
                {
                    random_model(12, 5, 3,  2),
                    random_model(12, 5, 10, 2)
                },
                {
                    require_optimality( /*tolerance*/ 1e-3),
                    require_optimization_status(OptimizationStatus::OPTIMAL)
                }
            );
            // clang-format on
        }
        SUBCASE("icm")
        {
            OptimizerParameters parameters;
            parameters["sub_optimizer"] = "icm";

            // clang-format off
            test_discrete_gm_optimizer(
                "reduced_gm_optimizer",
                parameters,
                {
                    random_model(24, 15, 3,  2),
                    random_model(24, 10, 10, 2)
                },
                // return status needs to be local_optimal OR optimal
                require_local_optimality(/*proven*/true)
            );
            // clang-format on
        }
    }
}

TEST_CASE("fusion_moves")
{
    SUBCASE("binary_models_full")
    {
        // when using a binary model, and use alpha expansion as proposal generator,
        // the first proposals will be all zeros, but since the starting point is also
        // all zeros, the fist fusion-move-model will have zero variables.
        // The second proposal will be all ones, and the second fusion-move-model will
        // have all variables -- The model is the same as the original model.
        //  This is a special case, which is tested here.

        // fusion parameters
        OptimizerParameters fusion_parameters;
        fusion_parameters["optimizer_name"] = "brute_force_naive";

        // fusion-moves parameters
        OptimizerParameters fusion_moves_parameters;
        fusion_moves_parameters["fusion_parameters"] = fusion_parameters;
        fusion_moves_parameters["proposal_gen_name"] = "alpha_expansion";
        fusion_moves_parameters["max_iterations"] = 2;

        // clang-format off
        test_discrete_gm_optimizer(
            "fusion_moves",
            fusion_moves_parameters,
            {
                potts_grid(3, 1, 2, true),
                random_sparse_model(8, 5, 1, 4, 2, 0.5 ),
                random_model(6, 6, 3, 2)
            },
            {
                require_optimization_status(OptimizationStatus::ITERATION_LIMIT_REACHED),
                require_optimality(/*tolerance*/ 1e-5)
            }
        );
        // clang-format on
    }

    SUBCASE("icm")
    {
        // We can emulate icm with fusion moves when
        // using proposals where just a single variable is changes.
        // Therefore, for a single variable we generate proposals for all labels
        // and do that for all variables, until no change are accepted for any variable.
        // This is very slow, but allows to test the fusion framework,
        // since we can test for the "local_optimal" status.
        // Ie. after optimizing with fusion moves, no single variable can be changed
        // to improve the solution. We  test this property in a brute force way.

        // fusion parameters
        OptimizerParameters fusion_parameters;
        fusion_parameters["optimizer_name"] = "brute_force_naive";

        // fusion-moves parameters
        OptimizerParameters fusion_moves_parameters;
        fusion_moves_parameters["fusion_parameters"] = fusion_parameters;
        fusion_moves_parameters["proposal_gen_name"] = "testing";
        fusion_moves_parameters["max_iterations"] = 0;
        OptimizerParameters proposal_gen_parameters;
        proposal_gen_parameters["name"] = "icm";
        fusion_moves_parameters["proposal_gen_parameters"] = proposal_gen_parameters;

        // clang-format off
        test_discrete_gm_optimizer(
            "fusion_moves",
            fusion_moves_parameters,
            {
                potts_grid(5, 5, 3, false),
                random_sparse_model(8, 5, 1, 4, 3, 0.5 ),
                random_model(6, 6, 3, 3),
                random_model(8, 6, 4, 3),
                unique_label_chain(8, 8),
            },
            {
                require_optimization_status(OptimizationStatus::CONVERGED),
                require_local_n_optimality(1)
            }
        );
        // clang-format on
    }

    SUBCASE("alpha_expansion")
    {
        // running alpha expansion with an optimal fusion moves optimizer
        // will give a local optimal solution.
        // ie no single variable can be changed to improve the solution.

        // fusion parameters
        OptimizerParameters fusion_parameters;
        fusion_parameters["optimizer_name"] = "brute_force_naive";

        // fusion-moves parameters
        OptimizerParameters fusion_moves_parameters;
        fusion_moves_parameters["fusion_parameters"] = fusion_parameters;
        fusion_moves_parameters["proposal_gen_name"] = "alpha_expansion";
        fusion_moves_parameters["max_iterations"] = 0;

        // clang-format off
        test_discrete_gm_optimizer(
            "fusion_moves",
            fusion_moves_parameters,
            {
                potts_grid(4, 2, 3, false),
                random_sparse_model(8, 5, 1, 4, 3, 0.5 ),
                random_model(6, 4, 3, 3)
            },
            {
                require_optimization_status(OptimizationStatus::CONVERGED),
                require_local_n_optimality(1)
            }
        );
        // clang-format on
    }
    SUBCASE("optimizer_based_higher_order")
    {
        // running with optimizer based proposal
        // gives us the guarantee to be not worse
        // than the optimizer itself. this is even true
        // when the fusion-optimizer is not optimal (ie icm)
        std::vector<std::string> fusion_optimizer_names = {"icm", "brute_force_naive", "hqpbo", "belief_propagation"};

        for (auto fusion_optimizer_name : fusion_optimizer_names)
        {
            SUBCASE(fusion_optimizer_name.c_str())
            {
                // fusion parameters
                OptimizerParameters fusion_parameters;
                fusion_parameters["optimizer_name"] = fusion_optimizer_name;

                // fusion-moves parameters
                OptimizerParameters fusion_moves_parameters;
                fusion_moves_parameters["fusion_parameters"] = fusion_parameters;
                fusion_moves_parameters["proposal_gen_name"] = "optimizer_based";

                OptimizerParameters proposal_gen_parameters;
                OptimizerParameters belief_propagation_parameters;
                belief_propagation_parameters["max_iterations"] = 10;
                belief_propagation_parameters["convergence_tolerance"] = 0.0001;
                belief_propagation_parameters["damping"] = 0.5;
                proposal_gen_parameters["optimizer_name"] = "belief_propagation";
                proposal_gen_parameters["optimizer_parameters"] = belief_propagation_parameters;

                fusion_moves_parameters["proposal_gen_parameters"] = proposal_gen_parameters;

                // clang-format off
                test_discrete_gm_optimizer(
                    "fusion_moves",
                    fusion_moves_parameters,
                    {
                        random_sparse_model(8, 5, 1, 4, 3, 0.5 ),
                        random_model(6, 4, 3, 3),
                        unique_label_chain(4, 4)
                    },
                    {
                        require_optimization_status(OptimizationStatus::CONVERGED),
                        require_not_worse_than("belief_propagation", belief_propagation_parameters)

                    }
                );
                // clang-format on
            }
        }
    }

    SUBCASE("optimizer_based_second_order")
    {
        // running with optimizer based proposal
        // gives us the guarantee to be not worse
        // than the optimizer itself. this is even true
        // when the fusion-optimizer is not optimal (ie icm)
        std::vector<std::string> fusion_optimizer_names = {"icm", "brute_force_naive", "hqpbo", "belief_propagation",
                                                           "qpbo"};

        for (auto fusion_optimizer_name : fusion_optimizer_names)
        {
            SUBCASE(fusion_optimizer_name.c_str())
            {
                // fusion parameters
                OptimizerParameters fusion_parameters;
                fusion_parameters["optimizer_name"] = fusion_optimizer_name;

                // fusion-moves parameters
                OptimizerParameters fusion_moves_parameters;
                fusion_moves_parameters["fusion_parameters"] = fusion_parameters;
                fusion_moves_parameters["proposal_gen_name"] = "optimizer_based";

                OptimizerParameters proposal_gen_parameters;
                OptimizerParameters belief_propagation_parameters;
                belief_propagation_parameters["max_iterations"] = 10;
                belief_propagation_parameters["convergence_tolerance"] = 0.0001;
                belief_propagation_parameters["damping"] = 0.5;
                proposal_gen_parameters["optimizer_name"] = "belief_propagation";
                proposal_gen_parameters["optimizer_parameters"] = belief_propagation_parameters;

                fusion_moves_parameters["proposal_gen_parameters"] = proposal_gen_parameters;

                // clang-format off
                test_discrete_gm_optimizer(
                    "fusion_moves",
                    fusion_moves_parameters,
                    {
                        potts_grid(4, 2, 3, false),
                        random_sparse_model(8, 8, 1, 2, 5, 0.5),
                        random_model(6, 4, 2, 2),
                        unique_label_chain(5, 5, true)
                    },
                    {
                        require_optimization_status(OptimizationStatus::CONVERGED),
                        require_not_worse_than("belief_propagation", belief_propagation_parameters)
                    }
                );
                // clang-format on
            }
        }
    }
}

} // namespace nxtgm
