#include <discrete_gm_optimizer_tester.hpp>
#include <nxtgm/optimizers/gm/discrete/qpbo.hpp>

TEST_CASE("qpbo")
{

    nxtgm::tests::test_discrete_gm_optimizer<nxtgm::Qpbo>(
        std::string("test-qpbo"), {nxtgm::Qpbo::parameters_type{}},
        std::make_tuple(nxtgm::tests::PottsChain{4, 2}, nxtgm::tests::PottsChain{7, 2}, nxtgm::tests::Star{3, 2},
                        nxtgm::tests::Star{5, 2}, nxtgm::tests::SparsePottsChain{4, 2}),
        100, std::make_tuple(nxtgm::tests::CheckOptimality{}));
}