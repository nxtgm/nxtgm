#include <nxtgm/models/gm/discrete_gm/testing/optimizer_tester.hpp>
#include <nxtgm/nxtgm_runtime_checks.hpp>
#include <nxtgm/optimizers/callbacks.hpp>
#include <sstream>

namespace nxtgm
{

template <class F>
void run_time_limited(F &&f, std::chrono::duration<double, std::milli> time_limit)
{
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start);

    // running mean of duration
    static std::chrono::duration<double, std::milli> duration_mean = duration;
    static std::size_t count = 1;

    while (true)
    {
        // check if we exceeded the next iteration
        // to be within the time limit
        auto now = std::chrono::high_resolution_clock::now();
        auto time_elapsed = std::chrono::duration<double, std::milli>(now - start);
        auto mean_duration = std::chrono::duration<double, std::milli>(time_elapsed / count);

        // std::cout<<"time elapsed: "<<time_elapsed.count()<<" ms"<<std::endl;
        if (time_elapsed + mean_duration > time_limit)
        {
            break;
        }
        f();
        ++count;
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::milli>(end - start);
}

std::pair<typename DiscreteGm::solution_type, SolutionValue> solve_brute_force(const DiscreteGm &model)
{

    using gm_type = std::decay_t<decltype(model)>;
    using solution_type = typename gm_type::solution_type;

    auto optimizer_parameters = OptimizerParameters();
    auto optimizer = nxtgm::discrete_gm_optimizer_factory(model, "brute_force_naive", optimizer_parameters);

    optimizer->optimize();
    return std::pair<solution_type, SolutionValue>(optimizer->best_solution(), optimizer->best_solution_value());
}

class TestReporterCallback : public DiscreteGmOptimizerBase::reporter_callback_base_type
{
  public:
    virtual ~TestReporterCallback() = default;
    using base_type = DiscreteGmOptimizerBase::reporter_callback_base_type;
    TestReporterCallback(const DiscreteGmOptimizerBase *optimizer, const std::string info);

    void begin() override;
    void end() override;
    bool report() override;

    bool called_begin_;
    bool called_end_;
    bool called_report_;
    SolutionValue last_best_value_;
    std::string info_;
};

TestReporterCallback::TestReporterCallback(const DiscreteGmOptimizerBase *optimizer, const std::string info)
    : base_type(optimizer),
      called_begin_(false),
      called_end_(false),
      called_report_(false),
      last_best_value_(),
      info_(info + std::string("TestReporterCallback check:"))
{
}

void TestReporterCallback::begin()
{

    const auto &model = optimizer()->model();
    auto best_solution = optimizer()->best_solution();

    NXTGM_TEST_OP(best_solution.size(), ==, model.num_variables(), info_ + "best_solution.size()");

    auto best_solution_value = model.evaluate(best_solution);

    NXTGM_TEST_EQ_TOL(best_solution_value.energy(), optimizer()->best_solution_value().energy(), 1e-6,
                      info_ + "best_solution_value");

    auto current_solution_value = model.evaluate(optimizer()->current_solution());
    NXTGM_CHECK_EQ_TOL(current_solution_value.energy(), optimizer()->current_solution_value().energy(), 1e-6,
                       info_ + "current_solution_value");

    // ensure best is not worse than current
    NXTGM_TEST_OP(optimizer()->best_solution_value().energy(), <=, optimizer()->current_solution_value().energy(),
                  info_ + " best is not worse than current");

    NXTGM_TEST(!called_begin_, "begin was not yet called");
    called_begin_ = true;
}

void TestReporterCallback::end()
{
    NXTGM_TEST(!called_end_, info_ + "called end");
    NXTGM_TEST(called_begin_, info_ + "called end");
    called_end_ = true;
}

bool TestReporterCallback::report()
{
    NXTGM_TEST(called_begin_, info_ + "called begin");
    NXTGM_TEST(!called_end_, info_ + "not called end");
    return true;
}

std::string RequireOptimizationStatus::name() const
{
    return "RequireOptimizationStatus";
}
void RequireOptimizationStatus::require(DiscreteGmOptimizerBase *optimizer, OptimizationStatus status,
                                        const std::string &info) const
{
    NXTGM_TEST(status == should_status, info + "status");
}
std::unique_ptr<DiscreteGmOptimizerRequireBase> require_optimization_status(OptimizationStatus should_status)
{
    return std::make_unique<RequireOptimizationStatus>(should_status);
}

RequireOptimizationStatus::RequireOptimizationStatus(OptimizationStatus should_status)
    : should_status(should_status)
{
}

RequireInfesibility::RequireInfesibility(bool proven)
    : proven(proven)
{
}

std::string RequireFeasiblity::name() const
{
    return "RequireFeasiblity";
}
void RequireFeasiblity::require(DiscreteGmOptimizerBase *optimizer, OptimizationStatus status,
                                const std::string &info) const
{
    const auto &model = optimizer->model();
    auto solution_value = model.evaluate(optimizer->best_solution(), false);
    NXTGM_TEST(solution_value.is_feasible(), info + "solution_value.is_feasible()");
}

std::unique_ptr<DiscreteGmOptimizerRequireBase> require_feasiblity()
{
    return std::make_unique<RequireFeasiblity>();
}

RequireOptimality::RequireOptimality(bool proven)
    : proven(proven)
{
}
std::unique_ptr<DiscreteGmOptimizerRequireBase> require_optimality(bool proven)
{
    return std::make_unique<RequireOptimality>(proven);
}

std::string RequireOptimality::name() const
{
    return "RequireOptimality";
}

void RequireOptimality::require(DiscreteGmOptimizerBase *optimizer, OptimizationStatus status,
                                const std::string &info) const
{
    if (proven)
    {
        NXTGM_TEST(status == OptimizationStatus::OPTIMAL, info);
    }
    const auto &model = optimizer->model();
    SolutionValue optimal_solution_value;
    discrete_solution optimal_solution;
    std::tie(optimal_solution, optimal_solution_value) = solve_brute_force(model);
    auto solution_value = model.evaluate(optimizer->best_solution(), false);

    NXTGM_TEST(optimal_solution_value.is_feasible() == solution_value.is_feasible(), info);
    if (optimal_solution_value.is_feasible())
    {

        auto print = [&]() {
            std::stringstream ss;
            ss << "optimal solution: " << std::endl;
            for (auto l : optimal_solution)
            {
                ss << l << " ";
            }
            ss << std::endl;
            ss << "        solution: " << std::endl;
            for (auto l : optimizer->best_solution())
            {
                ss << l << " ";
            }
            ss << std::endl;
            return ss.str();
        };

        // REQUIRE_MESSAGE(solution_value.energy() == doctest::Approx(optimal_solution_value.energy()), print());
        // REQUIRE(solution_value.energy() == doctest::Approx(optimal_solution_value.energy()));
        NXTGM_TEST_EQ_TOL(solution_value.energy(), optimal_solution_value.energy(), 1e-6, info + print());
    }
}

std::string RequireInfesibility::name() const
{
    return "RequireInfesibility";
}

void RequireInfesibility::require(DiscreteGmOptimizerBase *optimizer, OptimizationStatus status,
                                  const std::string &info) const
{
    if (proven)
    {
        NXTGM_TEST(status == OptimizationStatus::INFEASIBLE, info);
    }
    const auto &model = optimizer->model();
    auto solution_value = model.evaluate(optimizer->best_solution(), false);
    NXTGM_TEST(!solution_value.is_feasible(), info);
}
std::unique_ptr<DiscreteGmOptimizerRequireBase> require_infesibility(bool proven)
{
    return std::make_unique<RequireInfesibility>(proven);
}

RequireLocalOptimality::RequireLocalOptimality(bool proven)
    : proven(proven)
{
}

std::string RequireLocalOptimality::name() const
{
    return "RequireLocalOptimality";
}
std::unique_ptr<DiscreteGmOptimizerRequireBase> require_local_optimality(bool proven)
{
    return std::make_unique<RequireLocalOptimality>(proven);
}

void RequireLocalOptimality::require(DiscreteGmOptimizerBase *optimizer, OptimizationStatus status,
                                     const std::string &info) const
{

    if (proven)
    {
        NXTGM_TEST(status == OptimizationStatus::LOCAL_OPTIMAL, info);
    }

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

            auto print = [&]() {
                std::stringstream ss;
                ss << "could improve solution by chaging label of "
                      "variable "
                   << vi << " from " << l << " to " << li << std::endl;
                return ss.str();
            };

            // REQUIRE_MESSAGE(solution_value <= solution_copy_value, print());
            NXTGM_TEST_OP(solution_value, <=, solution_copy_value, info + print());

            // reset solution_copy
            solution_copy[vi] = l;
        }
    }
}

std::string RequireLocalNOptimality::name() const
{
    return "RequireLocalNOptimality";
}

RequireLocalNOptimality::RequireLocalNOptimality(std::size_t n)
    : n(n)
{
}

void RequireLocalNOptimality::require(DiscreteGmOptimizerBase *optimizer, OptimizationStatus status,
                                      const std::string &info) const
{
    if (n != 2 && n != 3)
    {
        throw std::runtime_error("only n=2 and n=3 supported");
    }
    // if (!optimizer->model().space().is_simple())
    // {
    //     throw std::runtime_error("only simple spaces supported");
    // }

    const auto &model = optimizer->model();
    const auto solution = optimizer->best_solution();
    auto solution_value = model.evaluate(solution);
    auto solution_copy = solution;

    const auto num_var = model.space().size();

    if (n == 2)
    {
        for (std::size_t v0 = 0; v0 < num_var; ++v0)
        {
            for (std::size_t v1 = 0; v1 < num_var; ++v1)
            {
                if (v0 == v1)
                {
                    continue;
                }

                for (discrete_label_type l0 = 0; l0 < model.space()[v0]; ++l0)
                {
                    for (discrete_label_type l1 = 0; l1 < model.space()[v1]; ++l1)
                    {

                        if (l0 == solution[v0] && l1 == solution[v1])
                        {
                            continue;
                        }

                        solution_copy[v0] = l0;
                        solution_copy[v1] = l1;
                        auto solution_copy_value = model.evaluate(solution_copy, false);

                        auto print = [&]() {
                            std::stringstream ss;
                            ss << "could improve solution by "
                                  "chaging labels of "
                                  "variables "
                               << v0 << " and " << v1 << " from " << solution[v0] << " and " << solution[v1] << " to "
                               << l0 << " and " << l1 << std::endl;
                            return ss.str();
                        };

                        NXTGM_TEST_OP(solution_value, <=, solution_copy_value, info + print());
                    }
                }
                // reset solution_copy
                solution_copy[v0] = solution[v0];
                solution_copy[v1] = solution[v1];
            }
        }
    }
    else if (n == 3)
    {
        for (std::size_t v0 = 0; v0 < num_var; ++v0)
        {
            for (std::size_t v1 = 0; v1 < num_var; ++v1)
            {
                for (std::size_t v2 = 0; v2 < num_var; ++v2)
                {
                    if (v0 == v1 || v0 == v2 || v1 == v2)
                    {
                        continue;
                    }

                    for (discrete_label_type l0 = 0; l0 < model.space()[v0]; ++l0)
                    {
                        for (discrete_label_type l1 = 0; l1 < model.space()[v1]; ++l1)
                        {
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

                                auto print = [&]() {
                                    std::stringstream ss;
                                    ss << "could improve solution by "
                                          "chaging labels of "
                                          "variables "
                                       << v0 << ", " << v1 << " and " << v2 << " from " << solution[v0] << ", "
                                       << solution[v1] << " and " << solution[v2] << " to " << l0 << ", " << l1
                                       << " and " << l2 << std::endl;
                                    return ss.str();
                                };

                                NXTGM_TEST_OP(solution_value, <=, solution_copy_value, info + print());
                            }
                        }
                    }
                    // reset solution_copy
                    solution_copy[v0] = solution[v0];
                    solution_copy[v1] = solution[v1];
                    solution_copy[v2] = solution[v2];
                }
            }
        }
    }
    else
    {
        throw std::runtime_error("only n=2 and n=3 supported");
    }
}

std::unique_ptr<DiscreteGmOptimizerRequireBase> require_local_n_optimality(std::size_t n)
{
    if (n == 1)
    {
        return std::make_unique<RequireLocalOptimality>();
    }
    else
    {
        return std::make_unique<RequireLocalNOptimality>(n);
    }
}

std::chrono::duration<double> TestDiscreteGmOptimizerOptions::default_per_model_time_limit()
{
    // check env var
    // NXTGM_TEST_PER_MODEL_TIME_LIMIT
    const double t = std::getenv("NXTGM_TEST_PER_MODEL_TIME_LIMIT") != nullptr
                         ? std::stod(std::getenv("NXTGM_TEST_PER_MODEL_TIME_LIMIT"))
                         : 5.0;

    return std::chrono::duration<double>(t);
};

void test_discrete_gm_optimizer(const std::string optimizer_name, const OptimizerParameters &parameters,
                                std::initializer_list<std::unique_ptr<DiscreteGmTestmodel>> model_generators,
                                std::initializer_list<std::unique_ptr<DiscreteGmOptimizerRequireBase>> requirements,
                                const TestDiscreteGmOptimizerOptions &options)
{

    for (const auto &model_gen : model_generators)
    {

        unsigned seed = options.seed;

        auto run = [&]() {
            auto gm_and_name = model_gen->operator()(seed);
            auto gm = std::move(gm_and_name.first);
            auto gm_name = std::move(gm_and_name.second);

            auto optimizer = discrete_gm_optimizer_factory(gm, optimizer_name, parameters);
            // INFO("Testing model instance ", gm_name.second, " with ", optimizer_name);

            // eval trivial sol
            discrete_solution sol(gm.num_variables(), 0);

            auto sol_value = gm.evaluate(sol);

            // run optimizer
            OptimizationStatus status;
            if (options.with_testing_callback)
            {
                std::string info = "\nmodel: " + gm_name + "\noptimizer: " + optimizer_name;
                TestReporterCallback reporter(optimizer.get(), info);
                status = optimizer->optimize(&reporter);
                NXTGM_TEST(reporter.called_begin_, "testing callback begin not called");
                NXTGM_TEST(reporter.called_end_, "testing callback end not called");
            }
            else
            {
                status = optimizer->optimize();
            }

            // check reuirements
            for (const auto &requirement : requirements)
            {
                std::string info =
                    "\nmodel: " + gm_name + "\noptimizer: " + optimizer_name + "\nrequirement: " + requirement->name();
                requirement->require(optimizer.get(), status, info);
            }
            ++seed;
        };

        auto budget = std::chrono::duration<double, std::milli>(options.per_model_time_limit);
        run_time_limited(run, budget);
    }
}

void test_discrete_gm_optimizer(const std::string optimizer_name, const OptimizerParameters &parameters,
                                std::unique_ptr<DiscreteGmTestmodel> model_generator,
                                std::initializer_list<std::unique_ptr<DiscreteGmOptimizerRequireBase>> requirements,
                                const TestDiscreteGmOptimizerOptions &options)
{
    test_discrete_gm_optimizer(optimizer_name, parameters, {std::move(model_generator)}, requirements, options);
}

void test_discrete_gm_optimizer(const std::string optimizer_name, const OptimizerParameters &parameters,
                                std::initializer_list<std::unique_ptr<DiscreteGmTestmodel>> model_generators,
                                std::unique_ptr<DiscreteGmOptimizerRequireBase> requirements,
                                const TestDiscreteGmOptimizerOptions &options)
{
    test_discrete_gm_optimizer(optimizer_name, parameters, model_generators, {std::move(requirements)}, options);
}

void test_discrete_gm_optimizer(const std::string optimizer_name, const OptimizerParameters &parameters,
                                std::unique_ptr<DiscreteGmTestmodel> model_generator,
                                std::unique_ptr<DiscreteGmOptimizerRequireBase> requirement,
                                const TestDiscreteGmOptimizerOptions &options)
{
    test_discrete_gm_optimizer(optimizer_name, parameters, {std::move(model_generator)}, {std::move(requirement)},
                               options);
}
} // namespace nxtgm
