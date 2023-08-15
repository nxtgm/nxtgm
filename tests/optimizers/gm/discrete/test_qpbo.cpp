#include <discrete_gm_optimizer_tester.hpp>
#include <nxtgm/optimizers/gm/discrete/qpbo.hpp>

TEST_CASE("qpbo")
{
    const auto submodular = true;
    // clang-format off
    nxtgm::tests::test_discrete_gm_optimizer(
        "qpbo",
        {
        },
        std::make_tuple(
            nxtgm::tests::PottsGrid{4, 4, 2, submodular},
            nxtgm::tests::PottsChain{4, 2},
            nxtgm::tests::PottsChain{7, 2},
            nxtgm::tests::Star{3, 2},
            nxtgm::tests::Star{5, 2},
            nxtgm::tests::SparsePottsChain{4, 2}
        ),
        100,
        std::make_tuple(nxtgm::tests::CheckOptimality{})
    );
    // clang-format on
}
