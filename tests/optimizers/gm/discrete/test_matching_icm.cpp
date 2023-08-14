#include <discrete_gm_optimizer_tester.hpp>
#include <nxtgm/optimizers/gm/discrete/matching_icm.hpp>

TEST_CASE("matching-icm")
{
    SUBCASE("n2")
    {
        njson params;
        params["subgraph_size"] = 2;

        nxtgm::tests::test_discrete_gm_optimizer<nxtgm::MatchingIcm>(
            std::string("test-matching-icm-n2"), {params},
            std::make_tuple(nxtgm::tests::UniqueLabelChain{2, 2}, nxtgm::tests::UniqueLabelChain{4, 4},
                            nxtgm::tests::UniqueLabelChain{8, 8}, nxtgm::tests::UniqueLabelChain{12, 12},
                            nxtgm::tests::UniqueLabelChain{2, 4}, nxtgm::tests::UniqueLabelChain{4, 8}),
            100, std::make_tuple(nxtgm::tests::CheckLocalNOptimality{2}));
    }
    SUBCASE("n3")
    {

        njson params;
        params["subgraph_size"] = 3;

        nxtgm::tests::test_discrete_gm_optimizer<nxtgm::MatchingIcm>(
            std::string("test-matching-icm-n3"), {params},
            std::make_tuple(

                nxtgm::tests::UniqueLabelChain{4, 4}, nxtgm::tests::UniqueLabelChain{6, 6},
                nxtgm::tests::UniqueLabelChain{4, 7}, nxtgm::tests::UniqueLabelChain{6, 12}),
            20,
            std::make_tuple(
                // 3 variables at a time
                nxtgm::tests::CheckLocalNOptimality{3}));
    }
}
