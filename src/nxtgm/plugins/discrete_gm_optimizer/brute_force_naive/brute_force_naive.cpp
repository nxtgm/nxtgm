#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>
#include <nxtgm/utils/timer.hpp>

namespace nxtgm
{

class BruteForceNaive : public DiscreteGmOptimizerBase
{
    // class parameters_type
    //   public:
    //     inline parameters_type(OptimizerParameters &&){
    //     }
    // };

  public:
    using base_type = DiscreteGmOptimizerBase;
    using solution_type = typename DiscreteGm::solution_type;

    using reporter_callback_wrapper_type = typename base_type::reporter_callback_wrapper_type;
    using repair_callback_wrapper_type = typename base_type::repair_callback_wrapper_type;

    inline static std::string name()
    {
        return "BruteForceNaive";
    }
    virtual ~BruteForceNaive() = default;

    BruteForceNaive(const DiscreteGm &gm, OptimizerParameters &&parameters);

    OptimizationStatus optimize_impl(reporter_callback_wrapper_type &, repair_callback_wrapper_type &,
                                     const_discrete_solution_span starting_point) override;

    SolutionValue best_solution_value() const override;
    SolutionValue current_solution_value() const override;

    const solution_type &best_solution() const override;
    const solution_type &current_solution() const override;

  private:
    // parameters_type parameters_;
    solution_type best_solution_;
    solution_type current_solution_;
    SolutionValue best_sol_value_;
    SolutionValue current_sol_value_;
};

class BruteForceFactory : public DiscreteGmOptimizerFactoryBase
{
  public:
    using factory_base_type = DiscreteGmOptimizerFactoryBase;
    virtual ~BruteForceFactory() = default;
    expected<std::unique_ptr<DiscreteGmOptimizerBase>> create(const DiscreteGm &gm,
                                                              OptimizerParameters &&params) const override
    {
        return std::make_unique<BruteForceNaive>(gm, std::move(params));
    }
    int priority() const override
    {
        return plugin_priority(PluginPriority::VERY_LOW);
    }
    std::string license() const override
    {
        return "MIT";
    }
    std::string description() const override
    {
        return "Naive brute force optimizer suitable for very small problems.";
    }
    OptimizerFlags flags() const override
    {
        return OptimizerFlags::OptimalOnTrees;
    }
};

BruteForceNaive::BruteForceNaive(const DiscreteGm &gm, OptimizerParameters &&parameters)
    : base_type(gm, parameters),
      best_solution_(gm.space().size(), 0),
      current_solution_(gm.space().size(), 0),
      best_sol_value_(),
      current_sol_value_()
{
    ensure_all_handled(name(), parameters);

    best_sol_value_ = gm.evaluate(best_solution_, false /* early exit when infeasible*/);
    current_sol_value_ = best_sol_value_;
}

} // namespace nxtgm

XPLUGIN_CREATE_XPLUGIN_FACTORY(nxtgm::BruteForceFactory);

namespace nxtgm
{

OptimizationStatus BruteForceNaive::optimize_impl(reporter_callback_wrapper_type &reporter_callback,
                                                  repair_callback_wrapper_type & /*repair_callback not used*/,
                                                  const_discrete_solution_span)
{

    OptimizationStatus status = OptimizationStatus::OPTIMAL;

    static const bool early_stop_infeasible = true;

    // just iterate over all possible solutions:
    // `exitable_for_each_solution` is like
    // `for_each_solution` but it can be exited
    // when the callback returns false.

    bool exit_via_callback = false;
    this->model().space().exitable_for_each_solution(current_solution_, [&](const solution_type &solution) {
        // evaluate the current solution
        this->current_sol_value_ = this->model().evaluate(solution, early_stop_infeasible);

        // if the current solution is better than the best solution
        if (this->current_sol_value_ < this->best_sol_value_)
        {
            this->best_sol_value_ = this->current_sol_value_;
            this->best_solution_ = solution;

            if (!this->report(reporter_callback))
            {
                status = OptimizationStatus::CALLBACK_EXIT;
                return false;
            }
        }

        // check if the time limit is reached
        if (this->time_limit_reached())
        {
            status = OptimizationStatus::TIME_LIMIT_REACHED;
            return false;
        }

        // continue the brtue force search
        return true;
    });

    if (!this->best_sol_value_.is_feasible())
    {
        status = OptimizationStatus::INFEASIBLE;
    }

    return status;
}

SolutionValue BruteForceNaive::best_solution_value() const
{
    return this->best_sol_value_;
}
SolutionValue BruteForceNaive::current_solution_value() const
{
    return this->current_sol_value_;
}

const typename BruteForceNaive::solution_type &BruteForceNaive::best_solution() const
{
    return this->best_solution_;
}
const typename BruteForceNaive::solution_type &BruteForceNaive::current_solution() const
{
    return this->current_solution_;
}

} // namespace nxtgm
