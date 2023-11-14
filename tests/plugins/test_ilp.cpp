#include <cmath>
#include <nxtgm/plugins/ilp/ilp_base.hpp>
#include <nxtgm/plugins/plugin_registry.hpp>
#include <random>
#include <test.hpp>
#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

namespace nxtgm
{

#ifdef _WIN32
#define SKIP_WIN doctest::skip(true)
#else
#define SKIP_WIN doctest::skip(false)
#endif

TEST_CASE("ilp" * SKIP_WIN)
{

    std::cout << "ilp test" << std::endl;
    std::size_t seed = 32;
    std::size_t num_var = 10;
    std::size_t num_labels = 100;
    bool with_ignore_label = true;
    with_ignore_label = with_ignore_label || num_labels < num_var;
    auto num_ilp_var = num_var * num_labels;

    std::cout << "ilp test 1" << std::endl;
    xt::random::seed(seed);

    xt::xtensor<energy_type, 2> tensor =
        xt::random::rand<energy_type>({num_var, num_labels}, energy_type(-1), energy_type(1));

    std::cout << "ilp test 2" << std::endl;
    IlpData ilp_data;

    ilp_data.add_variables(num_var * num_labels,
                           /*lower bound*/ 0,
                           /*upper bound*/ 1,
                           /*objective*/ 0.0,
                           /*is_integer*/ false);
    std::cout << "ilp test 3" << std::endl;
    // constraints that each variable has exactly one label
    for (std::size_t vi = 0; vi < num_var; ++vi)
    {
        ilp_data.begin_constraint(1, 1);
        for (std::size_t label = 0; label < num_labels; ++label)
        {
            ilp_data.add_constraint_coefficient(vi * num_labels + label, 1);
        }
    }
    std::cout << "ilp test 4" << std::endl;

    for (std::size_t label = 0; label < num_labels; ++label)
    {
        if (with_ignore_label && label == 0)
        {
            continue;
        }
        ilp_data.begin_constraint(0, 1);
        for (std::size_t vi = 0; vi < num_var; ++vi)
        {
            ilp_data.add_constraint_coefficient(vi * num_labels + label, 1);
        }
    }
    std::cout << "ilp test 5" << std::endl;
    for (auto i = 0; i < num_ilp_var; ++i)
    {
        ilp_data[i] = tensor[i];
    }
    std::cout << "ilp test 6" << std::endl;
    auto factory = get_plugin_registry<IlpFactoryBase>().get_factory("ilp_highs");
    std::cout << "ilp test 7" << std::endl;
    auto ilp_solver = factory->create(std::move(ilp_data), OptimizerParameters());
    std::cout << "ilp test 8" << std::endl;
    ilp_solver->optimize_lp();
    std::cout << "ilp test 9" << std::endl;
    std::vector<double> solution(num_var * num_labels);
    ilp_solver->get_solution(solution.data());
    std::cout << "ilp test 10" << std::endl;

    std::vector<double> per_label_sum(num_labels, 0.0);

    for (std::size_t vi = 0; vi < num_var; ++vi)
    {
        for (std::size_t label = 0; label < num_labels; ++label)
        {
            auto sol = solution[vi * num_labels + label];
            sol = sol < 0.0001 && sol > -0.0001 ? 0 : sol;
            per_label_sum[label] += sol;
        }
    }
    std::cout << "ilp test 11" << std::endl;

    // check that the solution is valid
    for (std::size_t vi = 0; vi < num_var; ++vi)
    {
        double sum = 0.0;
        for (std::size_t label = 0; label < num_labels; ++label)
        {
            sum += solution[vi * num_labels + label];
        }
        REQUIRE(sum == doctest::Approx(1.0));
    }
    std::cout << "ilp test 12" << std::endl;
    // check that per label sum is one
    for (std::size_t label = 0; label < num_labels; ++label)
    {
        if (with_ignore_label && label == 0)
        {
            continue;
        }
        REQUIRE(per_label_sum[label] >= 0);
        REQUIRE(per_label_sum[label] <= 1);
    }
    std::cout << "ilp test 13" << std::endl;
}

} // namespace nxtgm
