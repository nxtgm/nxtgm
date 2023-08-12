#include <discrete_gm_optimizer_tester.hpp>
#include <nxtgm/optimizers/gm/discrete/graph_cut.hpp>

TEST_CASE("graph-cut")
{
    const auto submodular = true;
    const auto num_labels = 2;
    // clang-format off
    nxtgm::tests::test_discrete_gm_optimizer<nxtgm::GraphCut>(
        std::string("test-graph-cut-small"),
        {
            nxtgm::GraphCut::parameters_type{}
        },
        std::make_tuple(
            nxtgm::tests::PottsGrid{3,  4, num_labels, submodular},
            nxtgm::tests::SubmodularGrid{3,  3},
            nxtgm::tests::PottsGrid{7,  2, num_labels, submodular},
            nxtgm::tests::PottsGrid{14, 1, num_labels, submodular},
            nxtgm::tests::PottsChain{10, num_labels, submodular}
        ),
        1000,
        std::make_tuple(
            nxtgm::tests::CheckOptimality{}
        )
    );
    nxtgm::tests::test_discrete_gm_optimizer<nxtgm::GraphCut>(
        std::string("test-graph-cut-large"),
        {
            nxtgm::GraphCut::parameters_type{}
        },
        std::make_tuple(
            nxtgm::tests::PottsGrid{4,  4, num_labels, submodular},
            nxtgm::tests::SubmodularGrid{4,  4},
            nxtgm::tests::PottsChain{16, num_labels, submodular}
        ),
        10,
        std::make_tuple(
            nxtgm::tests::CheckOptimality{}
        )
    );
    // clang-format on
}
