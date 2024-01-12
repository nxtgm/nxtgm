
#include <chrono>
#include <nxtgm/optimizers/gm/discrete/movemaker.hpp>
#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>
#include <queue>

#include <nxtgm/utils/timer.hpp>

#include <nxtgm/nxtgm.hpp>
#include <nxtgm/nxtgm_runtime_checks.hpp>
namespace nxtgm
{

class Icm : public DiscreteGmOptimizerBase
{

  public:
    using base_type = DiscreteGmOptimizerBase;
    using solution_type = typename DiscreteGm::solution_type;

    using reporter_callback_wrapper_type = typename base_type::reporter_callback_wrapper_type;
    using repair_callback_wrapper_type = typename base_type::repair_callback_wrapper_type;

    inline static std::string name()
    {
        return "Icm";
    }
    virtual ~Icm() = default;

    Icm(const DiscreteGm &gm, OptimizerParameters &&parameters);

    OptimizationStatus optimize_impl(reporter_callback_wrapper_type &, repair_callback_wrapper_type &,
                                     const_discrete_solution_span starting_point) override;

    SolutionValue best_solution_value() const override;
    SolutionValue current_solution_value() const override;

    const solution_type &best_solution() const override;
    const solution_type &current_solution() const override;

  private:
    void compute_labels();

    Movemaker movemaker_;
    std::vector<uint8_t> in_queue_;
    std::queue<std::size_t> queue_;
};

class IcmFactory : public DiscreteGmOptimizerFactoryBase
{
  public:
    using factory_base_type = DiscreteGmOptimizerFactoryBase;
    virtual ~IcmFactory() = default;
    expected<std::unique_ptr<DiscreteGmOptimizerBase>> create(const DiscreteGm &gm,
                                                              OptimizerParameters &&params) const override
    {
        // std::cout<<"IcmFactory::create"<<std::endl;
        auto icm = std::make_unique<Icm>(gm, std::move(params));
        // std::cout<<"IcmFactory::created"<<std::endl;
        return std::move(icm);
    }
    // std::unique_ptr<DiscreteGmOptimizerBase> create_me(const DiscreteGm &gm,
    //                                                           OptimizerParameters &&params) const override
    // {
    //     std::cout<<"IcmFactory::create"<<std::endl;
    //     auto icm =  std::make_unique<Icm>(gm, std::move(params));
    //     std::cout<<"IcmFactory::created"<<std::endl;
    //     return std::move(icm);
    // }
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
        return "Iterated conditional models optimizer";
    }
};

} // namespace nxtgm

XPLUGIN_CREATE_XPLUGIN_FACTORY(nxtgm::IcmFactory);

namespace nxtgm
{

Icm::Icm(const DiscreteGm &gm, OptimizerParameters &&parameters)
    : base_type(gm, parameters),
      movemaker_(gm),
      in_queue_(gm.num_variables(), 1)
{
    std::cout << "Icm::Icm" << std::endl;
    ensure_all_handled(name(), parameters);
    // put all variables on queue
    for (std::size_t i = 0; i < gm.num_variables(); ++i)
    {
        queue_.push(i);
    }
}

OptimizationStatus Icm::optimize_impl(reporter_callback_wrapper_type &reporter_callback,
                                      repair_callback_wrapper_type & /*repair_callback not used*/,
                                      const_discrete_solution_span starting_point)
{
    // set the starting point
    if (starting_point.size() > 0)
    {
        movemaker_.set_current_solution(starting_point);
    }

    // shortcut to the model
    const auto &gm = this->model();

    while (!queue_.empty())
    {

        // get next variable
        const auto vi = queue_.front();
        queue_.pop();
        in_queue_[vi] = 0;

        // move optimal
        const auto did_improve = movemaker_.move_optimal(vi);

        NXTGM_ASSERT_EQ_TOL(this->model().evaluate(this->best_solution()).energy(),
                            this->movemaker_.solution_value().energy(), 1e-6, "icm invariant failed (1)");
        NXTGM_ASSERT_EQ_TOL(this->model().evaluate(this->best_solution()).how_violated(),
                            this->movemaker_.solution_value().how_violated(), 1e-6, "icm invariant failed (1)");

        // if the solution improved we put all neighbors on the queue
        if (did_improve)
        {
            // report the current solution to callack
            if (!this->report(reporter_callback))
            {
                return OptimizationStatus::CALLBACK_EXIT;
            }

            // add all neighbors to the queue
            for (const auto &fi : movemaker_.factors_of_variables()[vi])
            {
                for (const auto &neighbour_vi : gm.factors()[fi].variables())
                {
                    if (neighbour_vi != vi && in_queue_[neighbour_vi] == 0)
                    {
                        queue_.push(neighbour_vi);
                        in_queue_[neighbour_vi] = 1;
                    }
                }
            }
            for (const auto &fi : movemaker_.constraints_of_variables()[vi])
            {
                for (const auto &neighbour_vi : gm.constraints()[fi].variables())
                {
                    if (neighbour_vi != vi && in_queue_[neighbour_vi] == 0)
                    {
                        queue_.push(neighbour_vi);
                        in_queue_[neighbour_vi] = 1;
                    }
                }
            }
        }

        // check if the time limit is reached
        if (this->time_limit_reached())
        {
            return OptimizationStatus::TIME_LIMIT_REACHED;
        }
    }

    return OptimizationStatus::LOCAL_OPTIMAL;
}

SolutionValue Icm::best_solution_value() const
{
    NXTGM_ASSERT_EQ_TOL(this->model().evaluate(this->best_solution()).energy(),
                        this->movemaker_.solution_value().energy(), 1e-6, "icm invariant failed");
    NXTGM_ASSERT_EQ_TOL(this->model().evaluate(this->best_solution()).how_violated(),
                        this->movemaker_.solution_value().how_violated(), 1e-6, "icm invariant failed");

    return this->movemaker_.solution_value();
}
SolutionValue Icm::current_solution_value() const
{
    return this->movemaker_.solution_value();
}

const typename Icm::solution_type &Icm::best_solution() const
{
    return this->movemaker_.solution();
}
const typename Icm::solution_type &Icm::current_solution() const
{
    return this->movemaker_.solution();
}

} // namespace nxtgm
