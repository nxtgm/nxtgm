#include <discrete_gm_optimizer_tester.hpp>
#include <nxtgm/optimizers/gm/discrete/brute_force_naive.hpp>

TEST_CASE("brute-force-naive"){
    

    nxtgm::tests::test_discrete_gm_optimizer<nxtgm::BruteForceNaive>(
        std::string("brute-force-naive"),
        {nxtgm::BruteForceNaive::parameter_type{}},
        std::make_tuple(
            nxtgm::tests::PottsChain{4, 2},
            nxtgm::tests::PottsChain{5, 3},
            nxtgm::tests::RandomModel{ /*nvar*/6 , /*nfac*/6, /*max arity*/3,   /*max label*/2},
            nxtgm::tests::RandomModel{ /*nvar*/10 , /*nfac*/6, /*max arity*/4,   /*max label*/3},
            nxtgm::tests::PottsChainWithLabelCosts{5, 5},
            nxtgm::tests::UniqueLabelChain{3, 4},
            nxtgm::tests::UniqueLabelChain{4, 5},
            nxtgm::tests::UniqueLabelChain{5, 5}
        ),
        500,
        std::make_tuple(
            nxtgm::tests::CheckFeasiblity{},
            // ::tests::CheckOptimality{}, we skip this because it would test against itself
            nxtgm::tests::CheckLocalOptimality{}
        )
    );
}