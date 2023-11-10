#pragma once
#include <nxtgm/functions/unique_labels_constraint_function.hpp>
#include <nxtgm/models/gm/discrete_gm.hpp>
#include <nxtgm/nxtgm.hpp>
#include <nxtgm/optimizers/optimizer_base.hpp>

namespace nxtgm
{

struct UseAll
{
    inline bool operator()(std::size_t) const
    {
        return true;
    }
};

template <class USE_FACTOR, class USE_CONSTRAINT>
class FilteredMovemaker
{
  public:
    inline FilteredMovemaker(const DiscreteGm &gm, const USE_FACTOR &use_factor, const USE_CONSTRAINT &use_constraint)
        : use_factor_(use_factor),
          use_constraint_(use_constraint),
          gm_(gm),
          current_solution_(gm.num_variables(), 0),
          current_solution_copy_(gm.num_variables(), 0),
          current_solution_value_(),
          factors_of_variables_(gm, use_factor),
          constraints_of_variables_(gm, use_constraint),
          max_num_labels_solution_value_buffer_(gm.space().max_num_labels(), SolutionValue(0, 0)),
          max_arity_labels_buffer_(gm.max_arity())
    {
        current_solution_value_ = gm.evaluate_if(current_solution_, false, use_factor_, use_constraint_);
    }

    template <class SOLUTION>
    inline void set_current_solution(SOLUTION &&solution)
    {
        current_solution_.assign(solution.begin(), solution.end());
        current_solution_value_ = gm_.evaluate_if(current_solution_, false, use_factor_, use_constraint_);
    }

    inline bool move_optimal(std::size_t variable)
    {
        const auto current_label = current_solution_[variable];
        const auto &factors = factors_of_variables_[variable];
        const auto &constraints = constraints_of_variables_[variable];

        const auto &factors_ids = factors_of_variables_[variable];
        const auto &constraints_ids = constraints_of_variables_[variable];

        const auto num_labels = gm_.num_labels(variable);

        // reset buffers
        for (discrete_label_type label = 0; label < num_labels; ++label)
        {
            max_num_labels_solution_value_buffer_[label] = SolutionValue(0, 0);
        }

        for (const auto fid : factors_ids)
        {
            const auto &factor = gm_.factors()[fid];
            factor.map_from_model(current_solution_, max_arity_labels_buffer_);
            const auto pos = factor.variable_position(variable);

            for (discrete_label_type l = 0; l < num_labels; ++l)
            {
                max_arity_labels_buffer_[pos] = l;
                const auto energy = factor(max_arity_labels_buffer_.data());
                max_num_labels_solution_value_buffer_[l] += SolutionValue(energy, 0);
            }
        }
        for (const auto cid : constraints_ids)
        {
            const auto &constraint = gm_.constraints()[cid];
            constraint.map_from_model(current_solution_, max_arity_labels_buffer_);
            const auto pos = constraint.variable_position(variable);

            for (discrete_label_type l = 0; l < num_labels; ++l)
            {
                max_arity_labels_buffer_[pos] = l;
                const auto how_violated = constraint(max_arity_labels_buffer_.data());
                max_num_labels_solution_value_buffer_[l] += SolutionValue(0, how_violated);
            }
        }

        // find argmin with stl in max_num_labels_solution_value_buffer_ vector
        auto min_value_iter = std::min_element(max_num_labels_solution_value_buffer_.begin(),
                                               max_num_labels_solution_value_buffer_.begin() + num_labels);

        if (min_value_iter == max_num_labels_solution_value_buffer_.end())
        {
            throw std::runtime_error("min_element failed");
        }
        const auto best_label = std::distance(max_num_labels_solution_value_buffer_.begin(), min_value_iter);

        if (best_label != current_label)
        {
            current_solution_[variable] = best_label;

            // update energy
            const auto new_energy = current_solution_value_.energy() -
                                    max_num_labels_solution_value_buffer_[current_label].energy() +
                                    max_num_labels_solution_value_buffer_[best_label].energy();
            const auto new_how_violated = current_solution_value_.how_violated() -
                                          max_num_labels_solution_value_buffer_[current_label].how_violated() +
                                          max_num_labels_solution_value_buffer_[best_label].how_violated();

            const auto new_solution_value = SolutionValue(new_energy, new_how_violated);
            current_solution_value_ = new_solution_value;

            return true;
        }

        return false;
    }

    inline const discrete_solution &solution() const
    {
        return current_solution_;
    }
    inline SolutionValue solution_value() const
    {
        return current_solution_value_;
    }

    inline const DiscreteGmFactorsOfVariables &factors_of_variables() const
    {
        return factors_of_variables_;
    }
    inline const DiscreteGmConstraintsOfVariables &constraints_of_variables() const
    {
        return constraints_of_variables_;
    }

    template <class F>
    void for_each_neighbour(std::size_t var, F &&f)
    {
        for (auto fi : factors_of_variables_[var])
        {
            for (auto factors_var : gm_.factors()[fi].variables())
            {
                if (factors_var != var)
                {
                    f(factors_var);
                }
            }
        }

        for (auto ci : constraints_of_variables_[var])
        {
            for (auto constraints_var : gm_.constraints()[ci].variables())
            {
                if (constraints_var != var)
                {
                    f(constraints_var);
                }
            }
        }
    }

  protected:
    USE_FACTOR use_factor_;
    USE_CONSTRAINT use_constraint_;

    const DiscreteGm &gm_;
    discrete_solution current_solution_;
    discrete_solution current_solution_copy_;
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
    inline Movemaker(const DiscreteGm &gm)
        : FilteredMovemaker<UseAll, UseAll>(gm, UseAll(), UseAll())
    {
    }
};

// filters are expected to be trivially copyable
class UseNoGlobalUniqueConstraints
{
  public:
    UseNoGlobalUniqueConstraints(const DiscreteGm &gm)
    {
        if (gm.max_constraint_arity() < gm.num_variables())
        {
            throw UnsupportedModelException("graphical model has no global unique label constraints");
        }
        for (auto ci = 0; ci < gm.num_constraints(); ++ci)
        {
            if (gm.constraints()[ci].arity() == gm.num_variables())
            {
                auto f = gm.constraints()[ci].function();
                // try dynamic cast to unique label constraint
                if (auto ulc = dynamic_cast<const UniqueLables *>(f))
                {
                    constraint_index_ = ci;
                    return;
                }
            }
        }
        throw UnsupportedModelException("graphical model has no global unique label constraints");
    }
    // copy constructor
    UseNoGlobalUniqueConstraints(const UseNoGlobalUniqueConstraints &other)
        : constraint_index_(other.constraint_index_)
    {
    }

    bool operator()(std::size_t constraint_index) const
    {
        return constraint_index != constraint_index_;
    }

  private:
    std::size_t constraint_index_;
};

struct SameLabelShape
{
    inline std::size_t operator[](std::size_t) const
    {
        return shape;
    }
    std::size_t shape;
};

class MatchingMovemaker : public FilteredMovemaker<UseAll, UseNoGlobalUniqueConstraints>
{
  public:
    inline MatchingMovemaker(const DiscreteGm &gm)
        : FilteredMovemaker<UseAll, UseNoGlobalUniqueConstraints>(gm, UseAll(), UseNoGlobalUniqueConstraints(gm))
    {
        // check if gm is usable
        if (!gm.space().is_simple())
        {
            throw UnsupportedModelException("MatchingMovemaker only works with simple spaces");
        }
        const auto num_labels = gm.space()[0];
        if (num_labels < gm.num_variables())
        {
            throw UnsupportedModelException("MatchingMovemaker only works with spaces with at least as "
                                            "many labels as variables");
        }

        // check if there are more labels than variables
        more_labels_as_variables_ = num_labels > gm.num_variables();

        // initialize unused labels
        if (more_labels_as_variables_)
        {
            for (auto lu = gm.num_variables(); lu < num_labels; ++lu)
            {
                unused_labels_.insert(lu);
            }
        }

        // iota as labels
        std::iota(current_solution_.begin(), current_solution_.end(), 0);
        current_solution_value_ = gm.evaluate(current_solution_);
    }

    template <class VARS>
    bool move_optimal(VARS &&vars)
    {
        bool improved = false;

        // helper function to evaluate a set of factors / constraints
        auto eval = [&](auto &&factors, auto &&constraints) -> SolutionValue {
            SolutionValue value(0, 0);
            for (auto fi : factors)
            {
                const auto &factor = gm_.factors()[fi];
                if (factor.arity() == 1)
                {
                    const auto current_label = current_solution_[factor.variables()[0]];
                    value.energy() += factor(&current_label);
                }
                else
                {
                    factor.map_from_model(current_solution_, max_arity_labels_buffer_);
                    value.energy() += factor(max_arity_labels_buffer_.data());
                }
            }

            // evaluate constraints
            for (auto ci : constraints)
            {
                const auto &constraint = gm_.constraints()[ci];
                // constraints are seldomly unary so no need for
                // an optimization here
                constraint.map_from_model(current_solution_, max_arity_labels_buffer_);
                value.how_violated() += constraint(max_arity_labels_buffer_.data());
            }
            return value;
        };

        const auto n_var = vars.size();

        // copy original solution to restore it later
        for (auto var : vars)
        {
            current_solution_copy_[var] = current_solution_[var];
        }

        // available labels are all those of vars + unused labels
        flat_set<discrete_label_type> available_labels = unused_labels_;
        for (auto var : vars)
        {
            available_labels.insert(current_solution_[var]);
        }

        // get a set of all factors and a set of all constraints
        // that are connected to any of the variables
        flat_set<std::size_t> factors;
        flat_set<std::size_t> constraints;

        for (auto var : vars)
        {
            for (auto fi : factors_of_variables_[var])
            {
                factors.insert(fi);
            }
            for (auto ci : constraints_of_variables_[var])
            {
                constraints.insert(ci);
            }
        }

        // energy of current solution for both variables
        // and the involved factors / constraints
        auto current_value = eval(factors, constraints);

        // iterate over all possible label combinations and evaluate
        // factors / constraints for each combination
        const auto num_available_labels = available_labels.size();

        auto for_each = [&](auto &&f) {
            if (n_var == 2)
            {

                for (auto li0 = 0; li0 < num_available_labels; ++li0)
                {
                    current_solution_[vars[0]] = available_labels.begin()[li0];
                    for (auto li1 = 0; li1 < num_available_labels; ++li1)
                    {
                        if (li0 == li1)
                        {
                            continue;
                        }
                        current_solution_[vars[1]] = available_labels.begin()[li1];
                        f();
                    }
                }
            }
            else if (n_var == 3)
            {
                for (auto li0 = 0; li0 < num_available_labels; ++li0)
                {
                    current_solution_[vars[0]] = available_labels.begin()[li0];
                    for (auto li1 = 0; li1 < num_available_labels; ++li1)
                    {
                        if (li0 == li1)
                        {
                            continue;
                        }
                        current_solution_[vars[1]] = available_labels.begin()[li1];

                        for (auto li2 = 0; li2 < num_available_labels; ++li2)
                        {
                            if (li0 == li2 || li1 == li2)
                            {
                                continue;
                            }
                            current_solution_[vars[2]] = available_labels.begin()[li2];
                            f();
                        }
                    }
                }
            }
            else if (n_var == 4)
            {
                for (auto li0 = 0; li0 < num_available_labels; ++li0)
                {
                    current_solution_[vars[0]] = available_labels.begin()[li0];
                    for (auto li1 = 0; li1 < num_available_labels; ++li1)
                    {
                        if (li0 == li1)
                        {
                            continue;
                        }
                        current_solution_[vars[1]] = available_labels.begin()[li1];

                        for (auto li2 = 0; li2 < num_available_labels; ++li2)
                        {
                            if (li0 == li2 || li1 == li2)
                            {
                                continue;
                            }
                            current_solution_[vars[2]] = available_labels.begin()[li2];
                            for (auto li3 = 0; li3 < num_available_labels; ++li3)
                            {
                                if (li0 == li3 || li1 == li3 || li2 == li3)
                                {
                                    continue;
                                }
                                current_solution_[vars[3]] = available_labels.begin()[li3];
                                f();
                            }
                        }
                    }
                }
            }
            else
            {
                throw std::runtime_error("not implemented yet");
            }
        };

        for_each([&]() {
            const auto val = eval(factors, constraints);
            if (val < current_value)
            {
                // update current solution copy to store best solution
                for (auto var : vars)
                {
                    current_solution_copy_[var] = current_solution_[var];
                }

                current_solution_value_ -= current_value;
                current_solution_value_ += val;
                current_value = val;
                improved = true;
            }
        });

        // for(auto li0=0; li0<num_available_labels; ++li0)
        // {
        //     current_solution_[vars[0]] = available_labels.begin()[li0];
        //     for(auto li1=0; li1<num_available_labels; ++li1)
        //     {
        //         if(li0 == li1)
        //         {
        //             continue;
        //         }
        //         current_solution_[vars[1]] = available_labels.begin()[li1];

        //         const auto val = eval(factors, constraints);
        //         if(val < current_value)
        //         {
        //             // update current solution copy to store best solution
        //             for(auto var : vars)
        //             {
        //                 current_solution_copy_[var] = current_solution_[var];
        //             }

        //             current_solution_value_ -= current_value;
        //             current_solution_value_ += val;
        //             current_value = val;
        //             improved =  true;
        //         }
        //     }
        // }

        // restore best known solution
        for (auto var : vars)
        {
            current_solution_[var] = current_solution_copy_[var];
        }
        if (improved && more_labels_as_variables_)
        {
            // update unused labels
            unused_labels_ = available_labels;

            for (auto var : vars)
            {
                unused_labels_.erase(current_solution_[var]);
            }
        }

        return improved;
    }

  private:
    bool more_labels_as_variables_;
    flat_set<discrete_label_type> unused_labels_;
};

} // namespace nxtgm
