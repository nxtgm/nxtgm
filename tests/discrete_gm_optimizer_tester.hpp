#pragma once

#include <test.hpp>
#include <testmodels.hpp>

#include <nxtgm/models/gm/discrete_gm.hpp>
#include <nxtgm/nxtgm.hpp>
#include <nxtgm/optimizers/callbacks.hpp>
#include <nxtgm/optimizers/gm/discrete/brute_force_naive.hpp>
#include <nxtgm/utils/tuple_for_each.hpp>

namespace nxtgm::tests
{

inline std::pair<typename DiscreteGm::solution_type, SolutionValue> solve_brute_force(const DiscreteGm &model)
{

    using gm_type = std::decay_t<decltype(model)>;
    using solution_type = typename gm_type::solution_type;
    using optimizer_type = nxtgm::BruteForceNaive;

    auto optimizer_parameters = typename optimizer_type::parameters_type();
    auto optimizer = std::make_unique<optimizer_type>(model, optimizer_parameters);

    optimizer->optimize();
    return std::pair<solution_type, SolutionValue>(optimizer->best_solution(), optimizer->best_solution_value());
}

class TestReporterCallback : public DiscreteGmOptimizerBase::reporter_callback_base_type
{
  public:
    using base_type = DiscreteGmOptimizerBase::reporter_callback_base_type;
    TestReporterCallback(const DiscreteGmOptimizerBase *optimizer)
        : base_type(optimizer),
          called_begin_(false),
          called_end_(false),
          called_report_(false)
    {
    }

    inline void begin() override
    {
        // check that the claimed best value is the same as evaluating the best
        // solution
        const auto &model = optimizer()->model();
        auto best_solution_value = model.evaluate(optimizer()->best_solution());
        CHECK(best_solution_value.energy() == doctest::Approx(optimizer()->best_solution_value().energy()));

        auto current_solution_value = model.evaluate(optimizer()->current_solution());
        CHECK(current_solution_value.energy() == doctest::Approx(optimizer()->current_solution_value().energy()));

        // ensure best is not worse than current
        CHECK(optimizer()->best_solution_value() <= optimizer()->current_solution_value());

        CHECK(!called_begin_);
        called_begin_ = true;
    }

    inline void end() override
    {
        CHECK(!called_end_);
        CHECK(called_begin_);
        called_end_ = true;
    }

    bool report() override
    {
        CHECK(called_begin_);
        CHECK(!called_end_);
        return true;
    }

    bool called_begin_;
    bool called_end_;
    bool called_report_;

    SolutionValue last_best_value_;
};

struct CheckOptimizationStatus
{
    static std::string name()
    {
        return "CheckOptimizationStatus";
    }
    void check(DiscreteGmOptimizerBase *optimizer, OptimizationStatus status) const
    {
        CHECK(status == should_status);
    }
    OptimizationStatus should_status;
};

struct CheckFeasiblity
{
    static std::string name()
    {
        return "CheckFeasiblity";
    }
    void check(DiscreteGmOptimizerBase *optimizer, OptimizationStatus status) const
    {
        const auto &model = optimizer->model();
        auto solution_value = model.evaluate(optimizer->best_solution(), false);
        CHECK(solution_value.is_feasible());
    }
};

struct CheckOptimality
{
    static std::string name()
    {
        return "CheckOptimality";
    }
    void check(DiscreteGmOptimizerBase *optimizer, OptimizationStatus status) const
    {
        if (proven)
        {
            CHECK(status == OptimizationStatus::OPTIMAL);
        }
        const auto &model = optimizer->model();
        SolutionValue optimal_solution_value;
        discrete_solution optimal_solution;
        std::tie(optimal_solution, optimal_solution_value) = solve_brute_force(model);
        auto solution_value = model.evaluate(optimizer->best_solution(), false);

        CHECK(optimal_solution_value.is_feasible() == solution_value.is_feasible());
        if (optimal_solution_value.is_feasible())
        {

            if (!CHECK(solution_value.energy() == doctest::Approx(optimal_solution_value.energy())))
            {
                std::cout << "optimal solution: " << std::endl;
                for (auto l : optimal_solution)
                {
                    std::cout << l << " ";
                }
                std::cout << std::endl;
                std::cout << "        solution: " << std::endl;
                for (auto l : optimizer->best_solution())
                {
                    std::cout << l << " ";
                }
                std::cout << std::endl;
            }
        }
    }
    bool proven = true;
};

struct CheckInfesibility
{
    static std::string name()
    {
        return "CheckInfesibility";
    }
    void check(DiscreteGmOptimizerBase *optimizer, OptimizationStatus status) const
    {
        if (proven)
        {
            CHECK(status == OptimizationStatus::INFEASIBLE);
        }
        const auto &model = optimizer->model();
        auto solution_value = model.evaluate(optimizer->best_solution(), false);
        CHECK(!solution_value.is_feasible());
    }
    bool proven = true;
};

struct CheckLocalOptimality
{
    static std::string name()
    {
        return "CheckLocalOptimality";
    }
    void check(DiscreteGmOptimizerBase *optimizer, OptimizationStatus status) const
    {
        const auto &model = optimizer->model();
        const auto solution = optimizer->best_solution();
        auto solution_value = model.evaluate(solution);
        auto solution_copy = solution;

        const auto num_var = model.space().size();
        for (std::size_t vi = 0; vi < num_var; ++vi)
        {
            const auto l = solution[vi];
            const auto num_labels = model.space()[vi];
            for (discrete_label_type li = 0; li < num_labels; ++li)
            {
                if (li == l)
                {
                    continue;
                }
                solution_copy[vi] = li;
                auto solution_copy_value = model.evaluate(solution_copy, false);
                if (!CHECK(solution_value <= solution_copy_value))
                {
                    std::cout << "could improve solution by chaging label of "
                                 "variable "
                              << vi << " from " << l << " to " << li << std::endl;
                }

                // reset solution_copy
                solution_copy[vi] = l;
            }
        }
    }
};

struct CheckLocalNOptimality
{
    static std::string name()
    {
        return "CheckLocalNOptimality";
    }
    void check(DiscreteGmOptimizerBase *optimizer, OptimizationStatus status) const
    {
        if (n != 2 && n != 3)
        {
            throw std::runtime_error("only n=2 and n=3 supported");
        }
        if (!optimizer->model().space().is_simple())
        {
            throw std::runtime_error("only simple spaces supported");
        }

        const auto &model = optimizer->model();
        const auto solution = optimizer->best_solution();
        auto solution_value = model.evaluate(solution);
        auto solution_copy = solution;

        const auto num_var = model.space().size();

        if (n == 2)
        {
            for (std::size_t v0 = 0; v0 < num_var; ++v0)
                for (std::size_t v1 = 0; v1 < num_var; ++v1)
                {
                    if (v0 == v1)
                    {
                        continue;
                    }

                    for (discrete_label_type l0 = 0; l0 < model.space()[v0]; ++l0)
                        for (discrete_label_type l1 = 0; l1 < model.space()[v1]; ++l1)
                        {

                            if (l0 == solution[v0] && l1 == solution[v1])
                            {
                                continue;
                            }

                            solution_copy[v0] = l0;
                            solution_copy[v1] = l1;
                            auto solution_copy_value = model.evaluate(solution_copy, false);
                            if (!CHECK(solution_value <= solution_copy_value))
                            {
                                std::cout << "could improve solution by "
                                             "chaging labels of "
                                             "variables "
                                          << v0 << " and " << v1 << " from " << solution[v0] << " and " << solution[v1]
                                          << " to " << l0 << " and " << l1 << std::endl;
                            }
                        }
                    // reset solution_copy
                    solution_copy[v0] = solution[v0];
                    solution_copy[v1] = solution[v1];
                }
        }
        else if (n == 3)
        {
            for (std::size_t v0 = 0; v0 < num_var; ++v0)
                for (std::size_t v1 = 0; v1 < num_var; ++v1)
                    for (std::size_t v2 = 0; v2 < num_var; ++v2)
                    {
                        if (v0 == v1 || v0 == v2 || v1 == v2)
                        {
                            continue;
                        }

                        for (discrete_label_type l0 = 0; l0 < model.space()[v0]; ++l0)
                            for (discrete_label_type l1 = 0; l1 < model.space()[v1]; ++l1)
                                for (discrete_label_type l2 = 0; l2 < model.space()[v2]; ++l2)
                                {

                                    if (l0 == solution[v0] && l1 == solution[v1] && l2 == solution[v2])
                                    {
                                        continue;
                                    }

                                    solution_copy[v0] = l0;
                                    solution_copy[v1] = l1;
                                    solution_copy[v2] = l2;
                                    auto solution_copy_value = model.evaluate(solution_copy, false);
                                    if (!CHECK(solution_value <= solution_copy_value))
                                    {
                                        std::cout << "could improve solution by "
                                                     "chaging labels of "
                                                     "variables "
                                                  << v0 << ", " << v1 << " and " << v2 << " from " << solution[v0]
                                                  << ", " << solution[v1] << " and " << solution[v2] << " to " << l0
                                                  << ", " << l1 << " and " << l2 << std::endl;
                                    }
                                }
                        // reset solution_copy
                        solution_copy[v0] = solution[v0];
                        solution_copy[v1] = solution[v1];
                        solution_copy[v2] = solution[v2];
                    }
        }
    }
    std::size_t n = 2;
};

template <class SOLVER_TYPE, class MODEL_GEN_TUPLE, class CHECKER_TUPLE>
void test_discrete_gm_optimizer(const std::string &testname, std::initializer_list<nlohmann::json> solver_parameters,
                                MODEL_GEN_TUPLE &&model_gen_tuple, std::size_t n_runs, CHECKER_TUPLE &&checker_tuple,
                                bool with_testing_callback = true)
{

    const std::size_t workload = n_runs * solver_parameters.size() * std::tuple_size_v<MODEL_GEN_TUPLE>;

    std::cout << testname << ":\n";
    nxtgm::tuple_for_each(model_gen_tuple, [&](auto &&model_gen) {
        for (auto i = 0; i < n_runs; ++i)
        {

            auto gen_result = model_gen();
            const DiscreteGm model = std::move(gen_result.first);
            const auto name = std::move(gen_result.second);
            INFO("Model Instance ", name);

            auto pi = 0;
            for (auto &&solver_parameter : solver_parameters)
            {
                auto solver = std::make_unique<SOLVER_TYPE>(model, solver_parameter);
                OptimizationStatus status;
                if (with_testing_callback)
                {
                    TestReporterCallback reporter(solver.get());
                    status = solver->optimize(&reporter);
                    CHECK(reporter.called_begin_);
                    CHECK(reporter.called_end_);
                }
                else
                {
                    status = solver->optimize();
                }

                nxtgm::tuple_for_each(checker_tuple, [&model, &solver, status](auto &&checker) {
                    // name of the type of the checker
                    using checker_type = std::decay_t<decltype(checker)>;
                    // INFO("Checker ", checker_type::name());

                    checker.check(solver.get(), status);
                });
                ++pi;
            }
        }
    });
    std::cout << "\n";
}

} // namespace nxtgm::tests
