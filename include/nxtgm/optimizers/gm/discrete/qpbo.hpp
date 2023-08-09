#pragma once

#include <chrono>
#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>

#include <nxtgm/plugins/qpbo_base.hpp>

namespace nxtgm
{

class Qpbo : public DiscreteGmOptimizerBase
{
  public:
    class parameters_type
    {
      public:
        std::chrono::duration<double> time_limit = std::chrono::duration<double>::max();
        std::string qpbo_plugin_name;
    };

    using base_type = DiscreteGmOptimizerBase;
    using solution_type = typename DiscreteGm::solution_type;

    using reporter_callback_wrapper_type = typename base_type::reporter_callback_wrapper_type;
    using repair_callback_wrapper_type = typename base_type::repair_callback_wrapper_type;

    using base_type::optimize;

    inline static std::string name()
    {
        return "Qpbo";
    }
    virtual ~Qpbo() = default;

    Qpbo(const DiscreteGm &gm, const parameters_type &parameters);

    OptimizationStatus optimize(reporter_callback_wrapper_type &, repair_callback_wrapper_type &,
                                const_discrete_solution_span starting_point) override;

    SolutionValue best_solution_value() const override;
    SolutionValue current_solution_value() const override;

    const solution_type &best_solution() const override;
    const solution_type &current_solution() const override;

    bool is_partial_optimal(std::size_t variable_index) const override;

  private:
    parameters_type parameters_;

    SolutionValue best_solution_value_;
    solution_type best_solution_;
    std::vector<int> qpbo_labels_;

    std::unique_ptr<QpboBase> qpbo_;
};
} // namespace nxtgm
