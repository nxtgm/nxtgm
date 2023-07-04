#pragma once

#include <nxtgm/models/gm/discrete_gm.hpp>
#include <nxtgm/optimizers/optimizer_base.hpp>

namespace nxtgm {
class Movemaker {
public:
  Movemaker(const DiscreteGm &gm, const discrete_solution &initial_solution);

  bool move_optimal(std::size_t variable);

  inline const discrete_solution &solution() const { return current_solution_; }
  inline SolutionValue solution_value() const {
    return current_solution_value_;
  }

  inline const DiscreteGmFactorsOfVariables &factors_of_variables() const {
    return factors_of_variables_;
  }
  inline const DiscreteGmConstraintsOfVariables &
  constraints_of_variables() const {
    return constraints_of_variables_;
  }

private:
  const DiscreteGm &gm_;
  discrete_solution current_solution_;
  SolutionValue current_solution_value_;
  DiscreteGmFactorsOfVariables factors_of_variables_;
  DiscreteGmConstraintsOfVariables constraints_of_variables_;

  // various buffers
  std::vector<SolutionValue> max_num_labels_solution_value_buffer_;
  std::vector<discrete_label_type> max_arity_labels_buffer_;
};
} // namespace nxtgm
