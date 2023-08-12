#pragma once

#include <chrono>
#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>

#include <nxtgm/plugins/min_st_cut_base.hpp>

namespace nxtgm
{

class GraphCut : public DiscreteGmOptimizerBase
{
  public:
    class parameters_type
    {
      public:
        std::chrono::duration<double> time_limit = std::chrono::duration<double>::max();
        std::string min_st_cut_plugin_name;
        double submodular_epsilon = 1e-6;
    };

    using base_type = DiscreteGmOptimizerBase;
    using solution_type = typename DiscreteGm::solution_type;

    using reporter_callback_wrapper_type = typename base_type::reporter_callback_wrapper_type;
    using repair_callback_wrapper_type = typename base_type::repair_callback_wrapper_type;

    using base_type::optimize;

    inline static std::string name()
    {
        return "GraphCut";
    }
    virtual ~GraphCut() = default;

    GraphCut(const DiscreteGm &gm, const parameters_type &parameters);

    OptimizationStatus optimize(reporter_callback_wrapper_type &, repair_callback_wrapper_type &,
                                const_discrete_solution_span starting_point) override;

    SolutionValue best_solution_value() const override;
    SolutionValue current_solution_value() const override;

    const solution_type &best_solution() const override;
    const solution_type &current_solution() const override;

  private:
    parameters_type parameters_;

    SolutionValue best_solution_value_;
    solution_type best_solution_;

    std::unique_ptr<MinStCutBase> min_st_cut_;
};
} // namespace nxtgm