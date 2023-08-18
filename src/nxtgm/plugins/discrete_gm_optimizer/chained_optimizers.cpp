#include <iostream>
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

    using base_type::optimize;

    inline static std::string name()
    {
        return "ChainedOptimizers";
    }
    virtual ~ChainedOptimizers() = default;

    ChainedOptimizers(const DiscreteGm &gm, const OptimizerParameters &parameters);

    OptimizationStatus optimize(reporter_callback_wrapper_type &, repair_callback_wrapper_type &,
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

NXTGM_OPTIMIZER_DEFAULT_FACTORY(ChainedOptimizers);
} // namespace nxtgm

XPLUGIN_CREATE_XPLUGIN_FACTORY(nxtgm::ChainedOptimizersDiscreteGmOptimizerFactory);

namespace nxtgm
{
ChainedOptimizers::ChainedOptimizers(const DiscreteGm &gm, const OptimizerParameters &parameters)
    : base_type(gm),
      parameters_(parameters),
      best_solution_value_(),
      best_solution_(gm.num_variables(), 0),
      is_partial_optimal_(gm.num_variables(), 0)
{

    best_solution_value_ = gm.evaluate(best_solution_, false /* early exit when infeasible*/);
}

OptimizationStatus ChainedOptimizers::optimize(reporter_callback_wrapper_type &reporter_callback,
                                               repair_callback_wrapper_type & /*repair_callback not used*/,
                                               const_discrete_solution_span)
{

    reporter_callback.begin();

    // start the timer
    AutoStartedTimer timer;

    // shortcut to the model
    const auto &gm = this->model();

    OptimizationStatus total_status = OptimizationStatus::CONVERGED;

    for (const auto &[optimizer_name, optimizer_parameters] : parameters_.optimizer_parameters)
    {
        auto optimizer = discrete_gm_optimizer_factory(gm, optimizer_name, optimizer_parameters);

        auto starting_point = const_discrete_solution_span(best_solution_.data(), best_solution_.size());
        auto status = optimizer->optimize(nullptr, nullptr, starting_point);

        auto solution = optimizer->best_solution();
        auto solution_value = gm.evaluate(solution, false /* early exit when infeasible*/);

        if (auto solution_value = optimizer->best_solution_value(); solution_value < best_solution_value_)
        {
            best_solution_value_ = solution_value;
            best_solution_ = optimizer->best_solution();

            reporter_callback.report();
        }

        if (status == OptimizationStatus::OPTIMAL)
        {
            total_status = OptimizationStatus::OPTIMAL;
            break;
        }
        else if (status == OptimizationStatus::INFEASIBLE)
        {
            total_status = OptimizationStatus::INFEASIBLE;
            break;
        }
        else if (status == OptimizationStatus::LOCAL_OPTIMAL)
        {
            total_status = OptimizationStatus::LOCAL_OPTIMAL;
        }
    }

    // indicate the end of the optimization
    reporter_callback.end();

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
