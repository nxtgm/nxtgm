#include <discrete_gm_optimizer_tester.hpp>

TEST_CASE("dynamic-programming")
{
    nxtgm::tests::test_discrete_gm_optimizer("dynamic_programming", {},
                                             std::make_tuple(nxtgm::tests::PottsChain{4, 2},
                                                             nxtgm::tests::PottsChain{7, 3},
                                                             nxtgm::tests::Star{3, 3}, // nxtgm::tests::Star{5, 3},
                                                             nxtgm::tests::SparsePottsChain{4, 4}),
                                             100, std::make_tuple(nxtgm::tests::CheckOptimality{}));
}
