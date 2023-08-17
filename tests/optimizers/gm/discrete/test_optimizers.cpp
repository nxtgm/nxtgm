#include <discrete_gm_optimizer_tester.hpp>

TEST_CASE("chained_optimizers")
{

    nxtgm::OptimizerParameters icm_params;
    icm_params["time_limit"] = 10000000;

    nxtgm::OptimizerParameters belief_propagation_params;
    belief_propagation_params["max_iterations"] = 100;
    belief_propagation_params["convergence_tolerance"] = 0.0001;
    belief_propagation_params["damping"] = 0.5;
    belief_propagation_params["normalize_messages"] = true;

    nxtgm::OptimizerParameters chained_optimizer_params;
    chained_optimizer_params["time_limit"] = 10000000;

    // order will be respected
    chained_optimizer_params["belief_propagation"] = belief_propagation_params;
    chained_optimizer_params["icm"] = icm_params;

    // clang-format off
    nxtgm::tests::test_discrete_gm_optimizer(
        "chained_optimizers", { chained_optimizer_params},

        std::make_tuple(
            nxtgm::tests::PottsChain{10, 2},
            nxtgm::tests::PottsChain{8, 3},
            nxtgm::tests::Star{10, 3},
            nxtgm::tests::RandomModel{/*nvar*/ 6, /*nfac*/ 6, /*max arity*/ 3,/*max label*/ 2},
            nxtgm::tests::RandomModel{/*nvar*/ 10, /*nfac*/ 6, /*max arity*/ 4, /*max label*/ 3}
        ),
        1000,
        std::make_tuple(
            nxtgm::tests::CheckOptimizationStatus{nxtgm::OptimizationStatus::LOCAL_OPTIMAL},
            nxtgm::tests::CheckLocalOptimality{}
        )
    );

    // clang-format on
}
