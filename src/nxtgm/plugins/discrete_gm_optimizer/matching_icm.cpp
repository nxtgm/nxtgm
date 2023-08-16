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
    class parameters_type : public OptimizerParametersBase
    {
      public:
        inline parameters_type(const nlohmann::json &json_parameters)
            : OptimizerParametersBase(json_parameters)
        {
            if (json_parameters.contains("subgraph_size"))
            {
                subgraph_size = json_parameters["subgraph_size"];
            }
        }
        int subgraph_size = 3;
    };

  public:
    using base_type = DiscreteGmOptimizerBase;
    using solution_type = typename DiscreteGm::solution_type;

    using reporter_callback_wrapper_type = typename base_type::reporter_callback_wrapper_type;
    using repair_callback_wrapper_type = typename base_type::repair_callback_wrapper_type;

    using base_type::optimize;

    inline static std::string name()
    {
        return "MatchingIcm";
    }
    virtual ~MatchingIcm() = default;

    MatchingIcm(const DiscreteGm &gm, const nlohmann::json &parameters);

    OptimizationStatus optimize(reporter_callback_wrapper_type &, repair_callback_wrapper_type &,
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

NXTGM_OPTIMIZER_DEFAULT_FACTORY(MatchingIcm);
} // namespace nxtgm

XPLUGIN_CREATE_XPLUGIN_FACTORY(nxtgm::MatchingIcmDiscreteGmOptimizerFactory);

namespace nxtgm
{
MatchingIcm::MatchingIcm(const DiscreteGm &gm, const nlohmann::json &json_parameters)
    : base_type(gm),
      parameters_(json_parameters),
      movemaker_(gm),
      in_queue_(gm.num_variables(), 1)
{
    // std::cout<<"CONSTRUCTING ICM"<<std::endl;
    //  put all variables on queue
    for (std::size_t i = 0; i < gm.num_variables(); ++i)
    {
        queue_.push(i);
    }
}

OptimizationStatus MatchingIcm::optimize(reporter_callback_wrapper_type &reporter_callback,
                                         repair_callback_wrapper_type & /*repair_callback not used*/,
                                         const_discrete_solution_span starting_point)
{
    if (starting_point.size() > 0)
    {
        this->movemaker_.set_current_solution(starting_point);
    }

    // indicate the start of the optimization
    reporter_callback.begin();

    // start the timer
    AutoStartedTimer timer;

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

                    if (reporter_callback && !timer.paused_call([&]() { return reporter_callback.report(); }))
                    {
                        status = OptimizationStatus::CALLBACK_EXIT;
                        return false;
                    }
                    // check if the time limit is reached
                    if (timer.elapsed() > this->parameters_.time_limit)
                    {
                        status = OptimizationStatus::TIME_LIMIT_REACHED;
                        return false;
                    }
                }
                return true;
            });
    }

    // indicate the end of the optimization
    reporter_callback.end();

    return OptimizationStatus::LOCAL_OPTIMAL;
}

SolutionValue MatchingIcm::best_solution_value() const
{
    // std::cout<<"claimed "<<this->movemaker_.solution_value()<<" ";
    // std::cout<<"actual
    // "<<this->model().evaluate(this->movemaker_.solution())<<std::endl;
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
