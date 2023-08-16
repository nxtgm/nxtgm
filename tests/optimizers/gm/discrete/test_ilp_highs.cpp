#include <discrete_gm_optimizer_tester.hpp>

TEST_CASE("ilp-highs" * SKIP_WIN)
{

    SUBCASE("ilp")
    {
        nxtgm::tests::test_discrete_gm_optimizer(
            "ilp_highs", {},
            std::make_tuple(nxtgm::tests::PottsChain{4, 2}, nxtgm::tests::PottsChain{7, 3}, nxtgm::tests::Star{5, 2},
                            nxtgm::tests::RandomModel{/*nvar*/ 6, /*nfac*/ 6,
                                                      /*max arity*/ 3,
                                                      /*max label*/ 2},
                            nxtgm::tests::RandomModel{/*nvar*/ 10, /*nfac*/ 6,
                                                      /*max arity*/ 4,
                                                      /*max label*/ 3},
                            nxtgm::tests::RandomSparseModel{/*nvar*/ 4, /*nfac*/ 3,
                                                            /*min arity*/ 2,
                                                            /*max arity*/ 4,
                                                            /*max label*/ 4,
                                                            /*density*/ 0.5},
                            nxtgm::tests::RandomSparseModel{/*nvar*/ 10, /*nfac*/ 10,
                                                            /*min arity*/ 2,
                                                            /*max arity*/ 3,
                                                            /*max label*/ 3,
                                                            /*density*/ 0.2},
                            nxtgm::tests::SparsePottsChain{5, 5}, nxtgm::tests::PottsChainWithLabelCosts{5, 5},
                            nxtgm::tests::UniqueLabelChain{2, 2, true}, nxtgm::tests::UniqueLabelChain{4, 5, true},
                            nxtgm::tests::UniqueLabelChain{5, 5, true}, nxtgm::tests::UniqueLabelChain{2, 2, false},
                            nxtgm::tests::UniqueLabelChain{4, 5, false}, nxtgm::tests::UniqueLabelChain{5, 5, false}),
            100, std::make_tuple(nxtgm::tests::CheckOptimality{}));
    }

    SUBCASE("infeasible")
    {
        nxtgm::tests::test_discrete_gm_optimizer(
            "ilp_highs", {}, std::make_tuple(nxtgm::tests::InfeasibleModel{4, 2}), 100,
            std::make_tuple(nxtgm::tests::CheckOptimizationStatus{nxtgm::OptimizationStatus::INFEASIBLE},
                            nxtgm::tests::CheckInfesibility{}));
    }
}
