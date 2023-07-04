#include <discrete_gm_optimizer_tester.hpp>
#include <nxtgm/optimizers/gm/discrete/icm.hpp>

TEST_CASE("icm") {

  nxtgm::tests::test_discrete_gm_optimizer<nxtgm::Icm>(
      std::string("test-icm"), {nxtgm::Icm::parameters_type{}},
      std::make_tuple(
          nxtgm::tests::PottsChain{10, 2}, nxtgm::tests::PottsChain{8, 3},
          nxtgm::tests::Star{10, 3},
          nxtgm::tests::RandomModel{/*nvar*/ 6, /*nfac*/ 6, /*max arity*/ 3,
                                    /*max label*/ 2},
          nxtgm::tests::RandomModel{/*nvar*/ 10, /*nfac*/ 6, /*max arity*/ 4,
                                    /*max label*/ 3},
          nxtgm::tests::PottsChainWithLabelCosts{5, 5},
          nxtgm::tests::UniqueLabelChain{3, 4},
          nxtgm::tests::UniqueLabelChain{4, 5},
          nxtgm::tests::UniqueLabelChain{5, 5},
          nxtgm::tests::UniqueLabelChain{10, 10},
          nxtgm::tests::InfeasibleModel{10, 2}),
      100,
      std::make_tuple(
          nxtgm::tests::CheckOptimizationStatus{
              nxtgm::OptimizationStatus::LOCAL_OPTIMAL},
          nxtgm::tests::CheckLocalOptimality{}));
}
