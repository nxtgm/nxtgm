#pragma once

#include <chrono>
#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>

namespace nxtgm {

class DynamicProgramming : public DiscreteGmOptimizerBase {
public:
  class parameters_type {
  public:
    std::vector<std::size_t> roots;
    std::chrono::duration<double> time_limit =
        std::chrono::duration<double>::max();
  };

  using base_type = DiscreteGmOptimizerBase;
  using solution_type = typename DiscreteGm::solution_type;

  using reporter_callback_wrapper_type =
      typename base_type::reporter_callback_wrapper_type;
  using repair_callback_wrapper_type =
      typename base_type::repair_callback_wrapper_type;

  using base_type::optimize;

  inline static std::string name() { return "DynamicProgramming"; }
  virtual ~DynamicProgramming() = default;

  DynamicProgramming(const DiscreteGm &gm, const parameters_type &parameters,
                     const solution_type &initial_solution = solution_type());

  OptimizationStatus optimize(reporter_callback_wrapper_type &,
                              repair_callback_wrapper_type &) override;

  SolutionValue best_solution_value() const override;
  SolutionValue current_solution_value() const override;

  const solution_type &best_solution() const override;
  const solution_type &current_solution() const override;

private:
  void compute_labels();

  parameters_type parameters_;
  solution_type best_solution_;
  SolutionValue best_sol_value_;

  DiscreteGmFactorsOfVariables factors_of_variables_;

  std::vector<energy_type> value_buffer_;
  std::vector<discrete_label_type> state_buffer_;
  std::vector<energy_type *> value_buffers_;
  std::vector<discrete_label_type *> state_buffers_;
  std::vector<std::size_t> node_order_;
  std::vector<std::size_t> ordered_nodes_;
};
} // namespace nxtgm
