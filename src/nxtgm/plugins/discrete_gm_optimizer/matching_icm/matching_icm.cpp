#include <chrono>
#include <nxtgm/optimizers/gm/discrete/movemaker.hpp>
#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>
#include <queue>
#include <vector>

#include <nxtgm/utils/timer.hpp>

#include <nxtgm/nxtgm.hpp>

namespace nxtgm
{

class MatchingIcm : public DiscreteGmOptimizerBase
{
    class parameters_type
    {
      public:
        inline parameters_type(OptimizerParameters &parameters)
        {
            parameters.assign_and_pop("subgraph_size", subgraph_size);
            ensure_all_handled(MatchingIcm::name(), parameters);
        }
        int subgraph_size = 3;
    };

  public:
    using base_type = DiscreteGmOptimizerBase;
    using solution_type = typename DiscreteGm::solution_type;

    using reporter_callback_wrapper_type = typename base_type::reporter_callback_wrapper_type;
    using repair_callback_wrapper_type = typename base_type::repair_callback_wrapper_type;

    inline static std::string name()
    {
        return "MatchingIcm";
    }
    virtual ~MatchingIcm() = default;

    MatchingIcm(const DiscreteGm &gm, OptimizerParameters &&parameters);

    OptimizationStatus optimize_impl(reporter_callback_wrapper_type &, repair_callback_wrapper_type &,
                                     const_discrete_solution_span starting_point) override;

    SolutionValue best_solution_value() const override;
    SolutionValue current_solution_value() const override;

    const solution_type &best_solution() const override;
    const solution_type &current_solution() const override;

  private:
    void compute_labels();

    parameters_type parameters_;

    MatchingMovemaker movemaker_;
    std::vector<std::size_t> in_queue_;
    std::queue<std::size_t> queue_;
};

class MatchingIcmFactory : public DiscreteGmOptimizerFactoryBase
{
  public:
    using factory_base_type = DiscreteGmOptimizerFactoryBase;
    virtual ~MatchingIcmFactory() = default;
    expected<std::unique_ptr<DiscreteGmOptimizerBase>> create(const DiscreteGm &gm,
                                                              OptimizerParameters &&params) const override
    {
        return std::make_unique<MatchingIcm>(gm, std::move(params));
    }
    int priority() const override
    {
        return plugin_priority(PluginPriority::LOW);
    }
    std::string license() const override
    {
        return "MIT";
    }
    std::string description() const override
    {
        return "Iterated conditional models optimizer for matching problems";
    }
    OptimizerFlags flags() const override
    {
        return OptimizerFlags::LocalOptimal;
    }
};

} // namespace nxtgm

XPLUGIN_CREATE_XPLUGIN_FACTORY(nxtgm::MatchingIcmFactory);

namespace nxtgm
{
MatchingIcm::MatchingIcm(const DiscreteGm &gm, OptimizerParameters &&parameters)
    : base_type(gm, parameters),
      parameters_(parameters),
      movemaker_(gm),
      in_queue_(gm.num_variables(), 1)
{

    for (std::size_t i = 0; i < gm.num_variables(); ++i)
    {
        queue_.push(i);
    }
}

OptimizationStatus MatchingIcm::optimize_impl(reporter_callback_wrapper_type &reporter_callback,
                                              repair_callback_wrapper_type & /*repair_callback not used*/,
                                              const_discrete_solution_span starting_point)
{
    if (starting_point.size() > 0)
    {
        this->movemaker_.set_current_solution(starting_point);
    }

    // shortcut to the model
    const auto &gm = this->model();

    OptimizationStatus status = OptimizationStatus::LOCAL_OPTIMAL;

    bool did_improve = true;
    while (did_improve)
    {
        did_improve = false;

        SameLabelShape shape{gm.num_variables()};
        std::vector<discrete_label_type> variables(parameters_.subgraph_size);
        exitable_n_nested_loop_unique_labels<discrete_label_type>(
            parameters_.subgraph_size, shape, variables, [&](const auto &_) {
                if (movemaker_.move_optimal(variables))
                {
                    did_improve = true;

                    if (!this->report(reporter_callback))
                    {
                        status = OptimizationStatus::CALLBACK_EXIT;
                        return false;
                    }
                    // check if the time limit is reached
                    if (this->time_limit_reached())
                    {
                        status = OptimizationStatus::TIME_LIMIT_REACHED;
                        return false;
                    }
                }
                return true;
            });
    }

    return OptimizationStatus::LOCAL_OPTIMAL;
}

SolutionValue MatchingIcm::best_solution_value() const
{
    return this->movemaker_.solution_value();
}
SolutionValue MatchingIcm::current_solution_value() const
{
    return this->movemaker_.solution_value();
}

const typename MatchingIcm::solution_type &MatchingIcm::best_solution() const
{
    return this->movemaker_.solution();
}
const typename MatchingIcm::solution_type &MatchingIcm::current_solution() const
{
    return this->movemaker_.solution();
}

} // namespace nxtgm
