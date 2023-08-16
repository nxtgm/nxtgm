
#include <chrono>
#include <nxtgm/optimizers/gm/discrete/movemaker.hpp>
#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>
#include <queue>

#include <nxtgm/utils/timer.hpp>

#include <nxtgm/nxtgm.hpp>

namespace nxtgm
{

class Icm : public DiscreteGmOptimizerBase
{
    class parameters_type : public OptimizerParametersBase
    {
      public:
        inline parameters_type(const nlohmann::json &json_parameters)
            : OptimizerParametersBase(json_parameters)
        {
        }
    };

  public:
    using base_type = DiscreteGmOptimizerBase;
    using solution_type = typename DiscreteGm::solution_type;

    using reporter_callback_wrapper_type = typename base_type::reporter_callback_wrapper_type;
    using repair_callback_wrapper_type = typename base_type::repair_callback_wrapper_type;

    using base_type::optimize;

    inline static std::string name()
    {
        return "Icm";
    }
    virtual ~Icm() = default;

    Icm(const DiscreteGm &gm, const nlohmann::json &parameters);

    OptimizationStatus optimize(reporter_callback_wrapper_type &, repair_callback_wrapper_type &,
                                const_discrete_solution_span starting_point) override;

    SolutionValue best_solution_value() const override;
    SolutionValue current_solution_value() const override;

    const solution_type &best_solution() const override;
    const solution_type &current_solution() const override;

  private:
    void compute_labels();

    parameters_type parameters_;

    Movemaker movemaker_;
    std::vector<uint8_t> in_queue_;
    std::queue<std::size_t> queue_;
};

NXTGM_OPTIMIZER_DEFAULT_FACTORY(Icm);

} // namespace nxtgm

XPLUGIN_CREATE_XPLUGIN_FACTORY(nxtgm::IcmDiscreteGmOptimizerFactory);

namespace nxtgm
{

Icm::Icm(const DiscreteGm &gm, const nlohmann::json &json_parameters)
    : base_type(gm),
      parameters_(json_parameters),
      movemaker_(gm),
      in_queue_(gm.num_variables(), 1)
{
    // put all variables on queue
    for (std::size_t i = 0; i < gm.num_variables(); ++i)
    {
        queue_.push(i);
    }
}

OptimizationStatus Icm::optimize(reporter_callback_wrapper_type &reporter_callback,
                                 repair_callback_wrapper_type & /*repair_callback not used*/,
                                 const_discrete_solution_span starting_point)
{
    // set the starting point
    if (starting_point.size() > 0)
    {
        movemaker_.set_current_solution(starting_point);
    }

    // indicate the start of the optimization
    reporter_callback.begin();

    // start the timer
    AutoStartedTimer timer;

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

        // if the solution improved we put all neighbors on the queue
        if (did_improve)
        {
            // report the current solution to callack
            if (reporter_callback && !timer.paused_call([&]() { return reporter_callback.report(); }))
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
        if (timer.elapsed() > this->parameters_.time_limit)
        {
            return OptimizationStatus::TIME_LIMIT_REACHED;
        }
    }

    // indicate the end of the optimization
    reporter_callback.end();

    return OptimizationStatus::LOCAL_OPTIMAL;
}

SolutionValue Icm::best_solution_value() const
{
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
