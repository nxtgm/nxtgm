#include <discrete_gm_optimizer_tester.hpp>
#include <nxtgm/optimizers/gm/discrete/belief_propagation.hpp>

TEST_CASE("belief-propagation")
{
    SUBCASE("trees")
    {

        njson parameters;
        parameters["max_iterations"] = 20000;

        nxtgm::tests::test_discrete_gm_optimizer(
            "belief_propagation", {parameters},
            std::make_tuple(nxtgm::tests::PottsChain{2, 2}, nxtgm::tests::PottsChain{12, 2},
                            nxtgm::tests::PottsChain{7, 4}, nxtgm::tests::Star{3, 3}, nxtgm::tests::Star{5, 3},
                            nxtgm::tests::SparsePottsChain{4, 4}),
            200,
            std::make_tuple(nxtgm::tests::CheckOptimality{false},
                            nxtgm::tests::CheckOptimizationStatus{nxtgm::OptimizationStatus::CONVERGED}));
    }
    SUBCASE("trees-with-damping")
    {

        njson parameters;
        parameters["max_iterations"] = 20000;
        parameters["damping"] = 0.9;

        nxtgm::tests::test_discrete_gm_optimizer(
            "belief_propagation", {parameters},
            std::make_tuple(nxtgm::tests::PottsChain{2, 2}, nxtgm::tests::PottsChain{12, 2},
                            nxtgm::tests::PottsChain{7, 4}, nxtgm::tests::Star{3, 3}, nxtgm::tests::Star{5, 3},
                            nxtgm::tests::SparsePottsChain{4, 4}),
            200,
            std::make_tuple(nxtgm::tests::CheckOptimality{false},
                            nxtgm::tests::CheckOptimizationStatus{nxtgm::OptimizationStatus::CONVERGED}));
    }
    // check for convergence on models to large to check for optimality
    SUBCASE("large-trees")
    {

        njson parameters;
        parameters["max_iterations"] = 20000;

        nxtgm::tests::test_discrete_gm_optimizer(
            "belief_propagation", {parameters},
            std::make_tuple(nxtgm::tests::PottsChain{200, 2}, nxtgm::tests::PottsChain{200, 4},
                            nxtgm::tests::Star{50, 3}, nxtgm::tests::SparsePottsChain{400, 4}),
            1000, std::make_tuple(nxtgm::tests::CheckOptimizationStatus{nxtgm::OptimizationStatus::CONVERGED}));
    }
}
