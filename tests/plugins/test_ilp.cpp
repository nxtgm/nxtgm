#include <cmath>
#include <nxtgm/plugins/ilp/ilp_base.hpp>
#include <nxtgm/plugins/plugin_registry.hpp>
#include <nxtgm_test_common.hpp>
#include <random>
#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include <nxtgm_test_common.hpp>

namespace nxtgm
{

#ifdef _WIN32
#define SKIP_WIN doctest::skip(true)
#else
#define SKIP_WIN doctest::skip(false)
#endif

TEST_CASE("ilp_lp" * SKIP_WIN)
{
    for (auto ilp_plugin : all_ilp_plugins())
    {
        SUBCASE(ilp_plugin.c_str())
        {
            std::size_t seed = 32;
            std::size_t num_var = 200;
            std::size_t num_labels = 200;
            bool with_ignore_label = true;
            with_ignore_label = with_ignore_label || num_labels < num_var;
            auto num_ilp_var = num_var * num_labels;

            xt::random::seed(seed);

            xt::xtensor<energy_type, 2> tensor =
                xt::random::rand<energy_type>({num_var, num_labels}, energy_type(-1), energy_type(1));

            IlpData ilp_data;

            ilp_data.add_variables(num_var * num_labels,
                                   /*lower bound*/ 0,
                                   /*upper bound*/ 1,
                                   /*objective*/ 0.0,
                                   /*is_integer*/ false);
            // constraints that each variable has exactly one label
            for (std::size_t vi = 0; vi < num_var; ++vi)
            {
                ilp_data.begin_constraint(1, 1);
                for (std::size_t label = 0; label < num_labels; ++label)
                {
                    ilp_data.add_constraint_coefficient(vi * num_labels + label, 1);
                }
            }

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
            for (auto i = 0; i < num_ilp_var; ++i)
            {
                ilp_data[i] = tensor[i];
            }
            OptimizerParameters parameters;
            parameters["integer"] = false;
            auto factory = get_plugin_registry<IlpFactoryBase>().get_factory(std::string("ilp_") + ilp_plugin);
            auto ilp_solver = factory->create(std::move(ilp_data), std::move(parameters));
            ilp_solver->optimize();
            std::vector<double> solution(num_var * num_labels);
            ilp_solver->get_solution(solution.data());

            std::vector<double> per_label_sum(num_labels, 0.0);

            for (std::size_t vi = 0; vi < num_var; ++vi)
            {
                for (std::size_t label = 0; label < num_labels; ++label)
                {
                    auto sol = solution[vi * num_labels + label];
                    REQUIRE(sol == doctest::Approx(std::round(sol)));
                    sol = sol < 0.0001 && sol > -0.0001 ? 0 : sol;
                    per_label_sum[label] += sol;
                }
            }

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
        }
    }
}

TEST_CASE("ilp_ilp" * SKIP_WIN)
{
    for (auto ilp_plugin : all_ilp_plugins())
    {
        SUBCASE(ilp_plugin.c_str())
        {
            std::size_t num_var = 6;
            std::size_t num_labels = 3;

            // num edges in fully connected graph
            std::size_t num_edges = num_var * (num_var - 1) / 2;

            // num variables in ilp
            std::size_t num_ilp_var = (num_var * num_labels) + (num_edges * num_labels * num_labels);

            IlpData ilp_data;

            ilp_data.add_variables(num_ilp_var,
                                   /*lower bound*/ 0,
                                   /*upper bound*/ 1,
                                   /*objective*/ 0.0,
                                   /*is_integer*/ true);

            xt::xtensor<energy_type, 1> tensor =
                xt::random::rand<energy_type>({num_var * num_labels}, energy_type(-1), energy_type(1));

            // add unary terms
            for (std::size_t i = 0; i < num_var * num_labels; ++i)
            {
                ilp_data[i] = tensor[i];
            }

            // constraints that each variable has exactly one label
            for (std::size_t vi = 0; vi < num_var; ++vi)
            {
                ilp_data.begin_constraint(1, 1);
                for (std::size_t label = 0; label < num_labels; ++label)
                {
                    ilp_data.add_constraint_coefficient(vi * num_labels + label, 1);
                }
            }

            std::size_t per_factor_num_var = num_labels * num_labels;

            // loop over all pairs
            auto factor_start = (num_var * num_labels);
            for (std::size_t vi = 0; vi < num_var - 1; ++vi)
            {
                for (std::size_t vj = vi + 1; vj < num_var; ++vj)
                {

                    // marginalization constraints and objective
                    ilp_data.begin_constraint(1, 1);
                    for (std::size_t label_i = 0; label_i < num_labels; ++label_i)
                    {
                        for (std::size_t label_j = 0; label_j < num_labels; ++label_j)
                        {
                            const auto var = factor_start + label_i * num_labels + label_j;
                            ilp_data.add_constraint_coefficient(var, 1);
                            // non submodular objective
                            ilp_data[var] = label_i == label_j ? 0 : -1;
                        }
                    }

                    std::vector<std::vector<std::vector<int>>> factor_var(2, std::vector<std::vector<int>>(num_labels));

                    for (std::size_t label_i = 0; label_i < num_labels; ++label_i)
                    {
                        for (std::size_t label_j = 0; label_j < num_labels; ++label_j)
                        {

                            factor_var[0][label_i].push_back(factor_start);
                            factor_var[1][label_j].push_back(factor_start);
                            ++factor_start;
                        }
                    }
                    // add marginalization constraints
                    for (std::size_t label_i = 0; label_i < num_labels; ++label_i)
                    {

                        ilp_data.begin_constraint(0, 0);
                        const auto var_0 = vi * num_labels + label_i;
                        ilp_data.add_constraint_coefficient(var_0, -1);
                        for (auto var : factor_var[0][label_i])
                        {
                            ilp_data.add_constraint_coefficient(var, 1);
                        }

                        ilp_data.begin_constraint(0, 0);
                        const auto var_1 = vj * num_labels + label_i;
                        ilp_data.add_constraint_coefficient(var_1, -1);
                        for (auto var : factor_var[1][label_i])
                        {
                            ilp_data.add_constraint_coefficient(var, 1);
                        }
                    }
                }
            }
            OptimizerParameters parameters;
            parameters["integer"] = true;
            auto factory = get_plugin_registry<IlpFactoryBase>().get_factory("ilp_" + ilp_plugin);
            auto ilp_solver = factory->create(std::move(ilp_data), std::move(parameters));
            ilp_solver->optimize();
            std::vector<double> solution(num_ilp_var);
            ilp_solver->get_solution(solution.data());

            const auto eps = 0.000000001;
            for (auto &sol : solution)
            {
                // check bounds
                REQUIRE(sol >= 0.0 - eps);
                REQUIRE(sol <= 1.0 + eps);

                // check integrality
                REQUIRE(sol == doctest::Approx(std::round(sol)));
            }

            // check that only one label is selected for each variable
            for (std::size_t var = 0; var < num_var; ++var)
            {
                auto sum = 0.0;
                for (std::size_t label = 0; label < num_labels; ++label)
                {
                    sum += solution[var * num_labels + label];
                }
                REQUIRE(sum == doctest::Approx(1.0));
            }

            // check that the factor variables sum to one
            // (ie that only one label is selected for each factor)
            factor_start = (num_var * num_labels);
            for (std::size_t vi = 0; vi < num_var - 1; ++vi)
            {
                for (std::size_t vj = vi + 1; vj < num_var; ++vj)
                {

                    auto sum = 0.0;
                    for (std::size_t label_i = 0; label_i < num_labels; ++label_i)
                    {
                        for (std::size_t label_j = 0; label_j < num_labels; ++label_j)
                        {
                            const auto var = factor_start + label_i * num_labels + label_j;
                            sum += solution[var];
                        }
                    }
                    REQUIRE(sum == doctest::Approx(1.0));
                    factor_start += per_factor_num_var;
                }
            }
        }
    }
}

} // namespace nxtgm
