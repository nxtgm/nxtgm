#include <test.hpp>

#include <vector>

#include <nxtgm/nxtgm.hpp>
#include <nxtgm/spaces/discrete_space.hpp>

#include <sstream>

namespace nxtgm
{

TEST_CASE("discrete-space")
{
    using solution_type = std::vector<nxtgm::discrete_label_type>;

    const std::size_t num_vars = 3;
    DiscreteSpace space(
        std::vector<discrete_label_type>{discrete_label_type(2), discrete_label_type(3), discrete_label_type(4)});

    auto solution = solution_type(space.size());

    std::vector<solution_type> solutions;
    std::vector<solution_type> solutions_shoulds;

    for (auto l0 = 0; l0 < space[0]; ++l0)
        for (auto l1 = 0; l1 < space[1]; ++l1)
            for (auto l2 = 0; l2 < space[2]; ++l2)
            {
                solution_type sol(space.size());
                sol[0] = l0;
                sol[1] = l1;
                sol[2] = l2;
                solutions_shoulds.push_back(sol);
            }

    space.for_each_solution(solution, [&](const solution_type &solution) { solutions.push_back(solution); });

    CHECK_EQ(solutions.size(), solutions_shoulds.size());
    for (std::size_t i = 0; i < solutions.size(); ++i)
    {
        CHECK_EQ(solutions[i], solutions_shoulds[i]);
    }
}

TEST_CASE("discrete-space-bind")
{
    SUBCASE("non-simple")
    {

        DiscreteSpace space(std::vector<discrete_label_type>{2, 3, 8, 4});

        std::vector<uint8_t> mask(space.size(), 0);

        mask[1] = 1;
        mask[3] = 1;

        auto [binded_space_, space_to_subspace_] = space.subspace(span<uint8_t>(mask.data(), mask.size()), true);

        DiscreteSpace binded_space = std::move(binded_space_);
        std::unordered_map<std::size_t, std::size_t> space_to_subspace = std::move(space_to_subspace_);

        // subspace has vi  [1 ,3]

        CHECK_EQ(binded_space.size(), 2);
        CHECK_EQ(binded_space[0], 3);
        CHECK_EQ(binded_space[1], 3);
        CHECK(!binded_space.is_simple());

        CHECK_EQ(space_to_subspace.size(), 2);
        CHECK(space_to_subspace.count(1));
        CHECK(space_to_subspace.count(3));

        CHECK(!space_to_subspace.count(0));
        CHECK(!space_to_subspace.count(2));

        CHECK_EQ(space_to_subspace[1], 0);
        CHECK_EQ(space_to_subspace[3], 1);
    }
    SUBCASE("simple")
    {
        DiscreteSpace space(10, 3);
    }
}

TEST_CASE("discrete-space-serialization")
{

    SUBCASE("non-simple")
    {
        const std::size_t num_vars = 3;
        DiscreteSpace space(
            std::vector<discrete_label_type>{discrete_label_type(2), discrete_label_type(3), discrete_label_type(4)});

        auto as_json = space.serialize_json();

        auto jspace = DiscreteSpace::deserialize_json(as_json);

        CHECK_EQ(space.is_simple(), jspace.is_simple());
        CHECK_EQ(space.size(), jspace.size());
        for (std::size_t i = 0; i < space.size(); ++i)
        {
            CHECK_EQ(space[i], jspace[i]);
        }
    }
    SUBCASE("simple")
    {
        DiscreteSpace space(10, 3);

        auto as_json = space.serialize_json();
        auto jspace = DiscreteSpace::deserialize_json(as_json);

        CHECK_EQ(space.is_simple(), jspace.is_simple());
        CHECK_EQ(space.size(), jspace.size());
        for (std::size_t i = 0; i < space.size(); ++i)
        {
            CHECK_EQ(space[i], jspace[i]);
        }
    }
}

} // namespace nxtgm
