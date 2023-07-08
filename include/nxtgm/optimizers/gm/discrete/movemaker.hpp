#pragma once

#include <nxtgm/models/gm/discrete_gm.hpp>
#include <nxtgm/optimizers/optimizer_base.hpp>

namespace nxtgm
{

struct UseAll
{
    inline bool operator()(std::size_t) const { return true; }
};

template <class USE_FACTOR, class USE_CONSTRAINT>
class FilteredMovemaker
{
public:
    inline FilteredMovemaker(const DiscreteGm& gm, const USE_FACTOR& use_factor,
                             const USE_CONSTRAINT& use_constraint)
        : use_factor_(use_factor), use_constraint_(use_constraint), gm_(gm),
          current_solution_(gm.num_variables(), 0), current_solution_value_(),
          factors_of_variables_(gm, use_factor),
          constraints_of_variables_(gm, use_constraint),
          max_num_labels_solution_value_buffer_(gm.space().max_num_labels()),
          max_arity_labels_buffer_(gm.max_arity())
    {
        current_solution_value_ = gm.evaluate_if(current_solution_, false,
                                                 use_factor_, use_constraint_);
    }

    template <class SOLUTION>
    inline void set_current_solution(SOLUTION&& solution)
    {
        current_solution_.assign(solution.begin(), solution.end());
        current_solution_value_ = gm_.evaluate_if(current_solution_, false,
                                                  use_factor_, use_constraint_);
    }

    inline bool move_optimal(std::size_t variable)
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
            max_num_labels_solution_value_buffer_[label] =
                SolutionValue(0, true);
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
            constraint.map_from_model(current_solution_,
                                      max_arity_labels_buffer_);
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
                max_num_labels_solution_value_buffer_[best_label]
                    .how_violated();

            const auto new_solution_value =
                SolutionValue(new_energy, new_how_violated);
            current_solution_value_ = new_solution_value;

            return true;
        }

        return false;
    }

    inline const discrete_solution& solution() const
    {
        return current_solution_;
    }
    inline SolutionValue solution_value() const
    {
        return current_solution_value_;
    }

    inline const DiscreteGmFactorsOfVariables& factors_of_variables() const
    {
        return factors_of_variables_;
    }
    inline const DiscreteGmConstraintsOfVariables&
    constraints_of_variables() const
    {
        return constraints_of_variables_;
    }

private:
    USE_FACTOR use_factor_;
    USE_CONSTRAINT use_constraint_;

    const DiscreteGm& gm_;
    discrete_solution current_solution_;
    SolutionValue current_solution_value_;
    DiscreteGmFactorsOfVariables factors_of_variables_;
    DiscreteGmConstraintsOfVariables constraints_of_variables_;

    // various buffers
    std::vector<SolutionValue> max_num_labels_solution_value_buffer_;
    std::vector<discrete_label_type> max_arity_labels_buffer_;
};

class Movemaker : public FilteredMovemaker<UseAll, UseAll>
{
public:
    inline Movemaker(const DiscreteGm& gm)
        : FilteredMovemaker<UseAll, UseAll>(gm, UseAll(), UseAll())
    {
    }
};

} // namespace nxtgm
