#include <test.hpp>

#include <nxtgm/models/solution_value.hpp>

TEST_CASE("test_solution_value")
{
    using sv = nxtgm::models::SolutionValue<double>;

    const auto feasible = sv(0, 1, 0);
    const auto infeasible = sv(0, 0, 0);

    CHECK(feasible.is_feasible() == true);
    CHECK(infeasible.is_feasible() == false);
    CHECK(feasible.is_feasible() != infeasible.is_feasible());

    CHECK(feasible < infeasible);
    CHECK(!(infeasible < feasible));
}
