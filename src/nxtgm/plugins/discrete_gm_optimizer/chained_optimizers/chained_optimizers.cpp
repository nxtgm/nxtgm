#include <nxtgm/nxtgm.hpp>
#include <nxtgm/optimizers/gm/discrete/discrete_gm_optimizer_factory.hpp>
#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>
#include <nxtgm/utils/timer.hpp>

namespace nxtgm
{

class ChainedOptimizers : public DiscreteGmOptimizerBase
{

  public:
    using base_type = DiscreteGmOptimizerBase;
    using solution_type = typename DiscreteGm::solution_type;

    using reporter_callback_wrapper_type = typename base_type::reporter_callback_wrapper_type;
    using repair_callback_wrapper_type = typename base_type::repair_callback_wrapper_type;

    inline static std::string name()
    {
        return "ChainedOptimizers";
    }
    virtual ~ChainedOptimizers() = default;

    ChainedOptimizers(const DiscreteGm &gm, OptimizerParameters &&parameters);

    OptimizationStatus optimize_impl(reporter_callback_wrapper_type &, repair_callback_wrapper_type &,
                                     const_discrete_solution_span starting_point) override;

    SolutionValue best_solution_value() const override;
    SolutionValue current_solution_value() const override;

    const solution_type &best_solution() const override;
    const solution_type &current_solution() const override;

    bool is_partial_optimal(std::size_t variable_index) const override;

  private:
    OptimizerParameters parameters_;

    SolutionValue best_solution_value_;
    solution_type best_solution_;
    std::vector<uint8_t> is_partial_optimal_;
};

class ChainedOptimizerFactory : public DiscreteGmOptimizerFactoryBase
{
  public:
    using factory_base_type = DiscreteGmOptimizerFactoryBase;
    virtual ~ChainedOptimizerFactory() = default;
    expected<std::unique_ptr<DiscreteGmOptimizerBase>> create(const DiscreteGm &gm,
                                                              OptimizerParameters &&params) const override
    {
        return std::make_unique<ChainedOptimizers>(gm, std::move(params));
    }
    int priority() const override
    {
        return plugin_priority(PluginPriority::MEDIUM);
    }
    std::string license() const override
    {
        return "MIT";
    }
    std::string description() const override
    {
        return "Chain multiple optizer st. optimzers are warm started with the best solution of the previous "
               "optimizer(s).";
    }
    OptimizerFlags flags() const override
    {
        return OptimizerFlags::MetaOptimizer | OptimizerFlags::WarmStartable;
    }
};

} // namespace nxtgm

XPLUGIN_CREATE_XPLUGIN_FACTORY(nxtgm::ChainedOptimizerFactory);

namespace nxtgm
{
ChainedOptimizers::ChainedOptimizers(const DiscreteGm &gm, OptimizerParameters &&parameters)
    : base_type(gm, parameters),
      parameters_(parameters),
      best_solution_value_(),
      best_solution_(gm.num_variables(), 0),
      is_partial_optimal_(gm.num_variables(), 0)
{
    parameters.optimizer_parameters.clear();
    ensure_all_handled(name(), parameters);
    best_solution_value_ = gm.evaluate(best_solution_, false /* early exit when infeasible*/);
}

OptimizationStatus ChainedOptimizers::optimize_impl(reporter_callback_wrapper_type &reporter_callback,
                                                    repair_callback_wrapper_type & /*repair_callback not used*/,
                                                    const_discrete_solution_span)
{
    // shortcut to the model
    const auto &gm = this->model();

    OptimizationStatus total_status = OptimizationStatus::CONVERGED;

    for (const auto &[optimizer_name, optimizer_parameters] : parameters_.optimizer_parameters)
    {
        auto expected_optimizer = discrete_gm_optimizer_factory(gm, optimizer_name, optimizer_parameters);
        if (!expected_optimizer)
        {
            throw std::runtime_error(expected_optimizer.error());
        }
        auto optimizer = std::move(expected_optimizer.value());

        auto starting_point = const_discrete_solution_span(best_solution_.data(), best_solution_.size());
        auto status = optimizer->optimize(nullptr, nullptr, starting_point);

        auto solution = optimizer->best_solution();
        auto solution_value = gm.evaluate(solution, false /* early exit when infeasible*/);

        if (auto solution_value = optimizer->best_solution_value(); solution_value < best_solution_value_)
        {
            best_solution_value_ = solution_value;
            best_solution_ = optimizer->best_solution();

            if (!this->report(reporter_callback))
            {
                return OptimizationStatus::CALLBACK_EXIT;
            }
        }
        if (this->time_limit_reached())
        {
            return OptimizationStatus::TIME_LIMIT_REACHED;
        }
        if (status == OptimizationStatus::OPTIMAL || status == OptimizationStatus::INFEASIBLE)
        {
            return status;
        }
        else if (status == OptimizationStatus::LOCAL_OPTIMAL)
        {
            total_status = status;
        }
    }

    return total_status;
}

SolutionValue ChainedOptimizers::best_solution_value() const
{
    return best_solution_value_;
}
SolutionValue ChainedOptimizers::current_solution_value() const
{
    return best_solution_value_;
}

const typename ChainedOptimizers::solution_type &ChainedOptimizers::best_solution() const
{
    return best_solution_;
}
const typename ChainedOptimizers::solution_type &ChainedOptimizers::current_solution() const
{
    return best_solution_;
}

bool ChainedOptimizers::is_partial_optimal(std::size_t variable_index) const
{
    return is_partial_optimal_[variable_index];
}

} // namespace nxtgm
