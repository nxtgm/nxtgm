#pragma once

#include <nxtgm/functions/discrete_constraint_function_base.hpp>
#include <nxtgm/functions/discrete_energy_function_base.hpp>
#include <nxtgm/models/solution_value.hpp>
#include <nxtgm/spaces/discrete_space.hpp>

#include <nlohmann/json.hpp>
#include <numeric>
#include <nxtgm/nxtgm.hpp>
#include <nxtgm/utils/serialize.hpp>
#include <vector>

namespace nxtgm
{

const_discrete_label_span local_solution_from_model_solution(const std::vector<std::size_t> &variables,
                                                             const span<const discrete_label_type> &solution,
                                                             std::vector<discrete_label_type> &local_labels_buffer);

// forward declaration
class DiscreteEnergyFunctionBase;

class DiscreteFactor
{
  public:
    template <class VARIABLES>
    DiscreteFactor(const VARIABLES &variables, std::size_t function_index, const DiscreteEnergyFunctionBase *function)
        : function_index_(function_index),
          function_(function),
          variables_(variables.begin(), variables.end())
    {
    }

    inline const DiscreteEnergyFunctionBase *function() const
    {
        return function_;
    }
    inline std::size_t function_index() const
    {
        return function_index_;
    }

    inline const std::vector<std::size_t> &variables() const
    {
        return variables_;
    }
    inline std::size_t variable(std::size_t variable_index) const
    {
        return variables_[variable_index];
    }
    const std::size_t arity() const
    {
        return variables_.size();
    }

    template <class MODEL_DATA, class FACTOR_DATA>
    void map_from_model(const MODEL_DATA &model_data, FACTOR_DATA &factor_data) const
    {
        const auto &variables = this->variables();
        for (std::size_t ai = 0; ai < variables.size(); ++ai)
        {
            factor_data[ai] = model_data[variables[ai]];
        }
    }

    inline energy_type operator()(const discrete_label_type *labels) const
    {
        return function_->value(labels);
    }
    inline energy_type operator()(std::initializer_list<discrete_label_type> labels) const
    {
        return function_->value(labels.begin());
    }
    inline void add_values(energy_type *energy) const
    {
        function_->add_values(energy);
    }
    inline void copy_values(energy_type *energy) const
    {
        function_->copy_values(energy);
    }

    std::size_t variable_position(std::size_t variable) const
    {
        const auto iter = std::find(variables_.begin(), variables_.end(), variable);
        if (iter == variables_.end())
        {
            return variables_.size();
        }
        return std::distance(variables_.begin(), iter);
    }

  private:
    std::size_t function_index_; // for serialization
    const DiscreteEnergyFunctionBase *function_;
    std::vector<std::size_t> variables_;
};

class DiscreteConstraint
{
  public:
    template <class VARIABLES>
    DiscreteConstraint(const VARIABLES &variables, const std::size_t function_index,
                       const DiscreteConstraintFunctionBase *function)
        : function_index_(function_index),
          function_(function),
          variables_(variables.begin(), variables.end())
    {
    }

    inline const DiscreteConstraintFunctionBase *function() const
    {
        return function_;
    }

    inline std::size_t function_index() const
    {
        return function_index_;
    }

    inline std::size_t variable(std::size_t variable_index) const
    {
        return variables_[variable_index];
    }
    inline const std::vector<std::size_t> &variables() const
    {
        return variables_;
    }

    const std::size_t arity() const
    {
        return variables_.size();
    }

    inline auto operator()(const discrete_label_type *labels) const
    {
        return function_->value(labels);
    }
    inline auto operator()(std::initializer_list<discrete_label_type> labels) const
    {
        return function_->value(labels.begin());
    }

    template <class MODEL_DATA, class CONSTRAINT_DATA>
    void map_from_model(const MODEL_DATA &model_data, CONSTRAINT_DATA &constraint_data) const
    {
        const auto &variables = this->variables();
        for (std::size_t ai = 0; ai < variables.size(); ++ai)
        {
            constraint_data[ai] = model_data[variables[ai]];
        }
    }
    std::size_t variable_position(std::size_t variable) const
    {
        const auto iter = std::find(variables_.begin(), variables_.end(), variable);
        if (iter == variables_.end())
        {
            return variables_.size();
        }
        return std::distance(variables_.begin(), iter);
    }

  private:
    std::size_t function_index_; // for serialization
    const DiscreteConstraintFunctionBase *function_;
    std::vector<std::size_t> variables_;
};

class DiscreteGm
{

  public:
    using solution_type = std::vector<discrete_label_type>;

    DiscreteGm(const DiscreteSpace &discrete_space);

    template <class NUM_LABELS_ITER>
    DiscreteGm(NUM_LABELS_ITER num_labels_begin, NUM_LABELS_ITER num_labels_end)
        : space_(num_labels_begin, num_labels_end),
          factors_(),
          energy_functions_(),
          constraints_(),
          constraint_functions_(),
          max_factor_arity_(0),
          max_constraint_arity_(0),
          max_factor_size_(0),
          max_constraint_size_(0)
    {
    }

    inline DiscreteGm(std::size_t num_var, discrete_label_type num_labels)
        : space_(num_var, num_labels),
          factors_(),
          energy_functions_(),
          constraints_(),
          constraint_functions_(),
          max_factor_arity_(0),
          max_constraint_arity_(0),
          max_factor_size_(0),
          max_constraint_size_(0)
    {
    }

    inline const DiscreteSpace &space() const
    {
        return space_;
    }

    inline const std::vector<DiscreteFactor> &factors() const
    {
        return factors_;
    }

    inline const std::vector<DiscreteConstraint> &constraints() const
    {
        return constraints_;
    }

    const DiscreteFactor &factor(std::size_t factor_index) const
    {
        return factors_[factor_index];
    }

    const DiscreteConstraint &constraint(std::size_t constraint_index) const
    {
        return constraints_[constraint_index];
    }

    inline std::size_t max_factor_arity() const
    {
        return max_factor_arity_;
    }

    inline std::size_t max_factor_size() const
    {
        return max_factor_size_;
    }

    inline std::size_t max_constraint_arity() const
    {
        return max_constraint_arity_;
    }

    inline std::size_t max_constraint_size() const
    {
        return max_constraint_size_;
    }

    inline std::size_t max_arity() const
    {
        // stupid windows
        return max_factor_arity_ > max_constraint_arity_ ? max_factor_arity_ : max_constraint_arity_;
    }

    inline discrete_label_type num_labels(std::size_t variable_index) const
    {
        return space_[variable_index];
    }
    inline std::size_t num_variables() const
    {
        return space_.size();
    }
    inline std::size_t num_factors() const
    {
        return factors_.size();
    }
    inline std::size_t num_constraints() const
    {
        return constraints_.size();
    }

    template <class F>
    void for_each_factor(F &&f) const
    {
        std::size_t i = 0;
        for (const auto &factor : factors_)
        {
            f(factor, i);
            ++i;
        }
    }

    template <class F>
    void for_each_higher_order_factor(F &&f) const
    {
        this->for_each_factor([&](const auto &factor, std::size_t i) {
            if (factor.variables().size() > 1)
            {
                f(factor, i);
            }
        });
    }
    template <class F>
    void for_each_unary_factor(F &&f) const
    {
        this->for_each_factor([&](const auto &factor, std::size_t i) {
            if (factor.variables().size() == 1)
            {
                f(factor, i);
            }
        });
    }

    template <class F>
    void for_each_constraint(F &&f) const
    {
        for (const auto &constraint : constraints_)
        {
            f(constraint);
        }
    }

    template <class F>
    void for_each_factor_and_constraint(F &&f) const
    {
        for (std::size_t i = 0; i < factors_.size(); ++i)
        {
            f(factors_[i], i, false);
        }
        for (std::size_t i = 0; i < constraints_.size(); ++i)
        {
            f(constraints_[i], i, true);
        }
    }

    std::size_t add_energy_function(std::unique_ptr<DiscreteEnergyFunctionBase> function);
    std::size_t add_constraint_function(std::unique_ptr<DiscreteConstraintFunctionBase> function);

    template <class DISCRETE_VARIABLES>
    std::size_t add_factor(DISCRETE_VARIABLES &&discrete_variables, std::size_t function_id)
    {
        // check if all variables are valid
        for (const auto &variable : discrete_variables)
        {
            if (variable >= space_.size())
            {
                throw std::runtime_error("Invalid variable index");
            }
        }
        // check if there are no duplicates
        for (std::size_t i = 0; i < discrete_variables.size(); ++i)
        {
            for (std::size_t j = i + 1; j < discrete_variables.size(); ++j)
            {
                if (discrete_variables[i] == discrete_variables[j])
                {
                    throw std::runtime_error("Duplicate variable index");
                }
            }
        }

        // check that the function matches the arity
        if (discrete_variables.size() != energy_functions_[function_id]->arity())
        {
            throw std::runtime_error("Function arity does not match the number of variables");
        }
        // check that the function shape matches the number of labels
        for (std::size_t i = 0; i < discrete_variables.size(); ++i)
        {
            if (space_[discrete_variables[i]] != energy_functions_[function_id]->shape(i))
            {
                std::stringstream ss;
                ss << "Function shape does not match the number of labels: " << space_[discrete_variables[i]]
                   << " != " << energy_functions_[function_id]->shape(i);
                throw std::runtime_error(ss.str());
            }
        }

        const std::size_t arity = discrete_variables.size();
        max_factor_arity_ = std::max(max_factor_arity_, arity);

        const auto size = shape_product(discrete_variables);
        max_factor_size_ = std::max(max_factor_size_, size);

        factors_.emplace_back(std::forward<DISCRETE_VARIABLES>(discrete_variables), function_id,
                              energy_functions_[function_id].get());
        return factors_.size() - 1;
    }

    template <class DISCRETE_VARIABLES>
    std::size_t add_constraint(DISCRETE_VARIABLES &&discrete_variables, std::size_t function_id)
    {

        // check if all variables are valid
        for (const auto &variable : discrete_variables)
        {
            if (variable >= space_.size())
            {
                throw std::runtime_error("Invalid variable index");
            }
        }
        // check if there are no duplicates
        for (std::size_t i = 0; i < discrete_variables.size(); ++i)
        {
            for (std::size_t j = i + 1; j < discrete_variables.size(); ++j)
            {
                if (discrete_variables[i] == discrete_variables[j])
                {
                    throw std::runtime_error("Duplicate variable index");
                }
            }
        }

        // check that the function matches the arity
        if (discrete_variables.size() != constraint_functions_[function_id]->arity())
        {
            throw std::runtime_error("Function arity does not match the number of variables");
        }
        // check that cid is valid
        if (function_id >= constraint_functions_.size())
        {
            throw std::runtime_error("Invalid constraint function id");
        }

        // check that the function shape matches the number of labels
        for (std::size_t i = 0; i < discrete_variables.size(); ++i)
        {
            if (space_[discrete_variables[i]] != constraint_functions_[function_id]->shape(i))
            {
                std::stringstream ss;
                ss << "Constraint shape does not match the number of labels: " << space_[discrete_variables[i]]
                   << " != " << constraint_functions_[function_id]->shape(i);
                throw std::runtime_error(ss.str());
            }
        }

        const std::size_t arity = discrete_variables.size();
        max_constraint_arity_ = std::max(max_constraint_arity_, arity);

        const auto size = shape_product(discrete_variables);
        max_constraint_size_ = std::max(max_constraint_size_, size);

        constraints_.emplace_back(std::forward<DISCRETE_VARIABLES>(discrete_variables), function_id,
                                  constraint_functions_[function_id].get());
        return constraints_.size() - 1;
    }

    template <class DISCRETE_VARIABLES_INDICES>
    std::size_t add_factor(std::initializer_list<DISCRETE_VARIABLES_INDICES> discrete_variables,
                           std::size_t function_id)
    {
        const std::size_t arity = discrete_variables.size();
        max_factor_arity_ = std::max(max_factor_arity_, arity);

        const auto size = shape_product(discrete_variables);
        max_factor_size_ = std::max(max_factor_size_, size);

        factors_.emplace_back(discrete_variables, function_id, energy_functions_[function_id].get());
        return factors_.size() - 1;
    }

    template <class DISCRETE_VARIABLES_INDICES>
    std::size_t add_constraint(std::initializer_list<DISCRETE_VARIABLES_INDICES> discrete_variables,
                               std::size_t function_id)
    {
        const std::size_t arity = discrete_variables.size();
        max_constraint_arity_ = std::max(max_constraint_arity_, arity);
        constraints_.emplace_back(discrete_variables, function_id, constraint_functions_[function_id].get());
        return factors_.size() - 1;
    }

    SolutionValue evaluate(const span<const discrete_label_type> &solution, bool early_stop_infeasible = false) const;
    SolutionValue evaluate(const solution_type &solution, bool early_stop_infeasible = false) const;
    SolutionValue evaluate(const discrete_label_type *solution, bool early_stop_infeasible = false) const;

    template <class USE_FACTOR, class USE_CONSTRAINT>
    SolutionValue evaluate_if(const span<const discrete_label_type> &solution, bool early_stop_infeasible,
                              USE_FACTOR &&use_factor, USE_CONSTRAINT &&use_constraint) const
    {
        bool total_is_feasible = true;
        energy_type total_how_violated = 0;

        // buffer holding the labels for the factors/constraints
        std::vector<discrete_label_type> local_labels_buffer(max_arity());

        std::size_t ci = 0;
        for (const auto &constraint : constraints_)
        {
            if (use_constraint(ci))
            {
                const const_discrete_label_span labels =
                    local_solution_from_model_solution(constraint.variables(), solution, local_labels_buffer);
                const auto how_violated = constraint.function()->value(labels.data());
                if (how_violated >= constraint_feasiblility_limit)
                {
                    if (early_stop_infeasible)
                    {
                        return SolutionValue{std::numeric_limits<energy_type>::infinity(), how_violated};
                    }
                    else
                    {
                        total_how_violated += how_violated;
                    }
                }
                else
                {
                    total_how_violated += how_violated;
                }
            }
            ++ci;
        }

        energy_type total_energy = 0;
        std::size_t fi = 0;
        for (const auto &factor : factors_)
        {
            if (use_factor(fi))
            {
                const const_discrete_label_span labels =
                    local_solution_from_model_solution(factor.variables(), solution, local_labels_buffer);
                total_energy += factor.function()->value(labels.data());
            }
            ++fi;
        }
        total_how_violated = total_how_violated < constraint_feasiblility_limit ? 0 : total_how_violated;
        return SolutionValue{total_energy, total_how_violated};
    }

    nlohmann::json serialize_json() const;
    static DiscreteGm deserialize_json(const nlohmann::json &json);

    void save_binary(const std::string &path) const;
    static DiscreteGm load_binary(const std::string &path);

    void serialize(Serializer &serializer) const;
    static DiscreteGm deserialize(Deserializer &deserializer);

    std::tuple<DiscreteGm, std::unordered_map<std::size_t, std::size_t>, SolutionValue> bind(
        span<const uint8_t>, span<const discrete_label_type> labeles, bool is_include_mask) const;

  private:
    template <class VARIABLE_INDICIES>
    std::size_t shape_product(VARIABLE_INDICIES &&variable_indicies) const
    {
        std::size_t product = 1;
        for (const auto &variable_index : variable_indicies)
        {
            product *= space_[variable_index];
        }
        return product;
    }

    DiscreteSpace space_;
    std::vector<DiscreteFactor> factors_;
    std::vector<std::unique_ptr<DiscreteEnergyFunctionBase>> energy_functions_;
    std::vector<DiscreteConstraint> constraints_;
    std::vector<std::unique_ptr<DiscreteConstraintFunctionBase>> constraint_functions_;
    std::size_t max_factor_arity_;
    std::size_t max_constraint_arity_;
    std::size_t max_factor_size_;
    std::size_t max_constraint_size_;
};

class DiscreteGmFactorsOfVariables : public std::vector<std::vector<std::size_t>>
{
  public:
    using base_type = std::vector<std::vector<std::size_t>>;

    template <class USE_FACTOR>
    inline DiscreteGmFactorsOfVariables(const DiscreteGm &gm, USE_FACTOR &&use_factor)
        : base_type(gm.space().size())
    {
        for (std::size_t fi = 0; fi < gm.factors().size(); ++fi)
        {
            if (use_factor(fi))
            {
                for (const auto &vi : gm.factors()[fi].variables())
                {
                    (*this)[vi].push_back(fi);
                }
            }
        }
    }

    inline DiscreteGmFactorsOfVariables(const DiscreteGm &gm)
        : DiscreteGmFactorsOfVariables(gm, [](std::size_t) { return true; })
    {
    }
};

class DiscreteGmConstraintsOfVariables : public std::vector<std::vector<std::size_t>>
{
  public:
    using base_type = std::vector<std::vector<std::size_t>>;

    template <class USE_CONSTRAINT>
    inline DiscreteGmConstraintsOfVariables(const DiscreteGm &gm, USE_CONSTRAINT &&use_constraint)
        : base_type(gm.space().size())
    {
        for (std::size_t fi = 0; fi < gm.constraints().size(); ++fi)
        {
            if (use_constraint(fi))
            {
                for (const auto &vi : gm.constraints()[fi].variables())
                {
                    (*this)[vi].push_back(fi);
                }
            }
        }
    }
    DiscreteGmConstraintsOfVariables(const DiscreteGm &gm)
        : DiscreteGmConstraintsOfVariables(gm, [](std::size_t) { return true; })
    {
    }
};

} // namespace nxtgm
