#include <discrete_gm_optimizer_tester.hpp>
#include <nxtgm/optimizers/gm/discrete/dynamic_programming.hpp>

TEST_CASE("dynamic-programming") {

  const bool integer_constraints = false;
  nxtgm::tests::test_discrete_gm_optimizer<nxtgm::DynamicProgramming>(
      std::string("test-dynamic-programming"),
      {nxtgm::DynamicProgramming::parameters_type{}},
      std::make_tuple(
          // nxtgm::tests::PottsChain{4, 2}//,
          // nxtgm::tests::PottsChain{7, 3},
          nxtgm::tests::Star{3, 3} // nxtgm::tests::Star{5, 3}
          ),
      1, std::make_tuple(nxtgm::tests::CheckOptimality{}));
}
