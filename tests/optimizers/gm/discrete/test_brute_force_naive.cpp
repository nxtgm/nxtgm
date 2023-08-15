#include <discrete_gm_optimizer_tester.hpp>
#include <nxtgm/optimizers/gm/discrete/brute_force_naive.hpp>

TEST_CASE("brute-force-naive")
{

    SUBCASE("basics")
    {
        nxtgm::tests::test_discrete_gm_optimizer(
            "brute_force_naive", {},
            std::make_tuple(nxtgm::tests::PottsChain{4, 2}, nxtgm::tests::PottsChain{5, 3}, nxtgm::tests::Star{6, 2},
                            nxtgm::tests::RandomModel{/*nvar*/ 6, /*nfac*/ 6,
                                                      /*max arity*/ 3,
                                                      /*max label*/ 2},
                            nxtgm::tests::RandomModel{/*nvar*/ 10, /*nfac*/ 6,
                                                      /*max arity*/ 4,
                                                      /*max label*/ 3},
                            nxtgm::tests::PottsChainWithLabelCosts{5, 5}, nxtgm::tests::UniqueLabelChain{3, 4},
                            nxtgm::tests::UniqueLabelChain{4, 5}, nxtgm::tests::SparsePottsChain{4, 4},
                            nxtgm::tests::UniqueLabelChain{5, 5}),
            500,
            std::make_tuple(nxtgm::tests::CheckOptimizationStatus{nxtgm::OptimizationStatus::OPTIMAL},
                            nxtgm::tests::CheckFeasiblity{},
                            // ::tests::CheckOptimality{}, we skip this because it would
                            // test against itself
                            nxtgm::tests::CheckLocalOptimality{}));
    };

    SUBCASE("time-limited")
    {

        njson parameters;
        // 50 milliseconds as seconds
        parameters["time_limit"] = 0.05;

        nxtgm::tests::test_discrete_gm_optimizer(
            "brute_force_naive", {parameters}, std::make_tuple(nxtgm::tests::PottsChain{10, 10}), 20,
            std::make_tuple(nxtgm::tests::CheckOptimizationStatus{nxtgm::OptimizationStatus::TIME_LIMIT_REACHED}));
    };

    SUBCASE("infeasible")
    {
        nxtgm::tests::test_discrete_gm_optimizer(
            "brute_force_naive", {}, std::make_tuple(nxtgm::tests::InfeasibleModel{4, 2}), 10,
            std::make_tuple(nxtgm::tests::CheckOptimizationStatus{nxtgm::OptimizationStatus::INFEASIBLE},
                            nxtgm::tests::CheckInfesibility{}));
    }
}
