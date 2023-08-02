#pragma once

#include <nxtgm/optimizers/gm/discrete/movemaker.hpp>
#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>

#include <chrono>
#include <queue>
#include <vector>

namespace nxtgm
{

class MatchingIcm : public DiscreteGmOptimizerBase
{
  public:
    class parameters_type
    {
      public:
        std::size_t subgraph_size = 2;
        std::chrono::duration<double> time_limit = std::chrono::duration<double>::max();
    };

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

    MatchingIcm(const DiscreteGm &gm, const parameters_type &parameters);

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
} // namespace nxtgm