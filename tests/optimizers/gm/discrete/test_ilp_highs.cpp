#include <discrete_gm_optimizer_tester.hpp>
#include <nxtgm/optimizers/gm/discrete/ilp_highs.hpp>

TEST_CASE("ilp-highs"){
    
    nxtgm::tests::test_discrete_gm_optimizer<nxtgm::IlpHighs>(
        std::string("test-ilp-highs"),
        {nxtgm::IlpHighs::parameter_type{}},
        std::make_tuple(
            nxtgm::tests::PottsChain{4, 2},
            nxtgm::tests::PottsChain{7, 3},
            nxtgm::tests::RandomModel{ /*nvar*/6 , /*nfac*/6, /*max arity*/3,   /*max label*/2},
            nxtgm::tests::RandomModel{ /*nvar*/10 , /*nfac*/6, /*max arity*/4,   /*max label*/3},
            nxtgm::tests::PottsChainWithLabelCosts{5, 5},
            nxtgm::tests::UniqueLabelChain{2, 2},
            nxtgm::tests::UniqueLabelChain{4, 5},
            nxtgm::tests::UniqueLabelChain{5, 5}
        ),
        100,
        std::make_tuple(
            nxtgm::tests::CheckOptimality{}
        )
    );
}