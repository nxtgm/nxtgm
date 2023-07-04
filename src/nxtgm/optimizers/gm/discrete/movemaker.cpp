#include <nxtgm/optimizers/gm/discrete/movemaker.hpp>

#include <iostream>

namespace nxtgm
{
Movemaker::Movemaker(const DiscreteGm& gm,
                     const discrete_solution& initial_solution)
    : gm_(gm), current_solution_(), current_solution_value_(),
      factors_of_variables_(gm), constraints_of_variables_(gm),
      max_num_labels_solution_value_buffer_(gm.space().max_num_labels()),
      max_arity_labels_buffer_(gm.max_arity())
{
    if (initial_solution.empty())
    {
        current_solution_.resize(gm.num_variables());
    }
    else
    {
        current_solution_ = initial_solution;
    }
    current_solution_value_ = gm.evaluate(current_solution_);
}

bool Movemaker::move_optimal(std::size_t variable)
{
    const auto current_label = current_solution_[variable];
    const auto& factors = factors_of_variables_[variable];
    const auto& constraints = constraints_of_variables_[variable];

    const auto& factors_ids = factors_of_variables_[variable];
    const auto& constraints_ids = constraints_of_variables_[variable];

    const auto num_labels = gm_.num_labels(variable);

    // reset buffers
    for (discrete_label_type label = 0; label < num_labels; ++label)
    {
        max_num_labels_solution_value_buffer_[label] = SolutionValue(0, true);
    }

    for (const auto fid : factors_ids)
    {
        const auto& factor = gm_.factors()[fid];
        factor.map_from_model(current_solution_, max_arity_labels_buffer_);
        const auto pos = factor.variable_position(variable);

        for (discrete_label_type l = 0; l < num_labels; ++l)
        {
            max_arity_labels_buffer_[pos] = l;
            const auto energy = factor(max_arity_labels_buffer_.data());
            max_num_labels_solution_value_buffer_[l] +=
                SolutionValue(energy, 0);
        }
    }
    for (const auto cid : constraints_ids)
    {
        const auto& constraint = gm_.constraints()[cid];
        constraint.map_from_model(current_solution_, max_arity_labels_buffer_);
        const auto pos = constraint.variable_position(variable);

        for (discrete_label_type l = 0; l < num_labels; ++l)
        {
            max_arity_labels_buffer_[pos] = l;
            const auto how_violated =
                constraint(max_arity_labels_buffer_.data());
            max_num_labels_solution_value_buffer_[l] +=
                SolutionValue(0, how_violated);
        }
    }

    // find argmin with stl in max_num_labels_solution_value_buffer_ vector
    auto min_value_iter = std::min_element(
        max_num_labels_solution_value_buffer_.begin(),
        max_num_labels_solution_value_buffer_.begin() + num_labels);
    const auto best_label = std::distance(
        max_num_labels_solution_value_buffer_.begin(), min_value_iter);

    if (best_label != current_label)
    {
        current_solution_[variable] = best_label;

        // update energy
        const auto new_energy =
            current_solution_value_.energy() -
            max_num_labels_solution_value_buffer_[current_label].energy() +
            max_num_labels_solution_value_buffer_[best_label].energy();
        const auto new_how_violated =
            current_solution_value_.how_violated() -
            max_num_labels_solution_value_buffer_[current_label]
                .how_violated() +
            max_num_labels_solution_value_buffer_[best_label].how_violated();

        const auto new_solution_value =
            SolutionValue(new_energy, new_how_violated);
        current_solution_value_ = new_solution_value;

        return true;
    }

    return false;
}
} // namespace nxtgm
