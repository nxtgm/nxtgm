#include <test.hpp>
#include <testmodels.hpp>

#include <math.h>

#include <nxtgm/models/gm/discrete_gm.hpp>
#include <nxtgm/spaces/discrete_space.hpp>

TEST_CASE("test discrete gm")
{

    using value_type = float;
    using discrete_label_type = uint8_t;
    constexpr discrete_label_type n_states = 2;
    using discrete_space_type = nxtgm::space::discrete::StaticSimpleSpace<discrete_label_type, n_states - 1>;
    using discrete_gm_type = nxtgm::models::gm::discrete::Gm<value_type, discrete_space_type>;

    // binary simple chain model
    auto n_var = 4;

    auto space = discrete_space_type(n_var);
    auto gm = discrete_gm_type(space);

    CHECK_EQ(gm.space().size(), n_var);
    CHECK_EQ(gm.space().upper_bound(0), n_states - 1);
    CHECK_EQ(gm.space().upper_bound(1), n_states - 1);
    CHECK_EQ(gm.space().upper_bound(2), n_states - 1);
    CHECK_EQ(gm.space().upper_bound(3), n_states - 1);
}

TEST_CASE("test unique label chain")
{
    auto model = nxtgm::tests::unique_label_chain<float, uint8_t>(3, 4);
    std::cout << "model space " << model.space().size() << std::endl;
    using gm_type = std::decay_t<decltype(model)>;

    using solution_type = typename gm_type::space_type::solution_type;
    auto solution = solution_type(model.space());

    auto solution_value = model(solution, false);
    CHECK_EQ(solution_value.is_feasible(), false);

    solution[0] = 0;
    solution[1] = 0;
    solution[2] = 0;
    CHECK_EQ(model(solution, false).is_feasible(), false);

    solution[0] = 0;
    solution[1] = 1;
    solution[2] = 2;
    CHECK_EQ(model(solution, false).is_feasible(), true);

    solution[0] = 0;
    solution[1] = 1;
    solution[2] = 1;
    CHECK_EQ(model(solution, false).is_feasible(), false);
}
