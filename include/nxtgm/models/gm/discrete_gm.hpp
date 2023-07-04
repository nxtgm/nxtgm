#pragma once

#include <nxtgm/energy_functions/discrete_energy_function_base.hpp>
#include <nxtgm/constraint_functions/discrete_constraint_function_base.hpp>
#include <nxtgm/models/solution_value.hpp>
#include <nxtgm/spaces/discrete_space.hpp>

#include <nxtgm/nxtgm.hpp>

#include <vector>
#include <iostream>
#include <numeric>
#include <nlohmann/json.hpp>

namespace nxtgm
{

    const_discrete_label_span local_solution_from_model_solution(
        const std::vector<std::size_t> & variables,
        const std::vector<discrete_label_type> & solution,
        std::vector<discrete_label_type> & local_labels_buffer
    );


    class DiscreteFactor
    {
    public:


        template <class VARIABLES>
        DiscreteFactor(const VARIABLES &variables, const DiscreteEnergyFunctionBase *function)
        : function_(function),
          variables_(variables.begin(), variables.end())
        {
        }

        inline const DiscreteEnergyFunctionBase *function() const
        {
            return function_;
        }

        inline const std::vector<std::size_t> &variables() const
        {
            return variables_;
        }
        const std::size_t arity()const
        {
            return variables_.size();
        }

        template<class MODEL_DATA, class FACTOR_DATA>
        void map_from_model(const MODEL_DATA & model_data, FACTOR_DATA & factor_data)const
        {
            const auto & variables = this->variables();
            for(std::size_t ai = 0; ai < variables.size(); ++ai)
            {
                factor_data[ai] = model_data[variables[ai]];
            }
        }

        inline energy_type operator()(const discrete_label_type *labels)const
        {
            return function_->energy(labels);
        }
        inline energy_type operator()(std::initializer_list<discrete_label_type> labels)const
        {
            return function_->energy(labels.begin());
        }
        inline void add_energies(energy_type * energy, discrete_label_type *labels)const
        {
            function_->add_energies(energy, labels);
        }

        std::size_t variable_position(std::size_t variable)const
        {
            const auto iter = std::find(variables_.begin(), variables_.end(), variable);
            if(iter == variables_.end())
            {
                return variables_.size();
            }
            return std::distance(variables_.begin(), iter);
        }
    private:
        const DiscreteEnergyFunctionBase *function_;
        std::vector<std::size_t> variables_;
    };


    class DiscreteConstraint
    {
    public:

        template <class VARIABLES>
        DiscreteConstraint(const VARIABLES &variables, const DiscreteConstraintFunctionBase *function)
        : function_(function),
          variables_(variables.begin(), variables.end())
        {
        }

        inline const DiscreteConstraintFunctionBase *function() const
        {
            return function_;
        }

        inline const std::vector<std::size_t> &variables() const
        {
            return variables_;
        }

        const std::size_t arity()const
        {
            return variables_.size();
        }


        inline auto operator()(const discrete_label_type *labels)const
        {
            return function_->how_violated(labels);
        }
        inline auto operator()(std::initializer_list<discrete_label_type> labels)const
        {
            return function_->how_violated(labels.begin());
        }


        template<class MODEL_DATA, class CONSTRAINT_DATA>
        void map_from_model(const MODEL_DATA & model_data, CONSTRAINT_DATA & constraint_data)const
        {
            const auto & variables = this->variables();
            for(std::size_t ai = 0; ai < variables.size(); ++ai)
            {
                constraint_data[ai] = model_data[variables[ai]];
            }
        }
        std::size_t variable_position(std::size_t variable)const
        {
            const auto iter = std::find(variables_.begin(), variables_.end(), variable);
            if(iter == variables_.end())
            {
                return variables_.size();
            }
            return std::distance(variables_.begin(), iter);
        }
    private:
        const DiscreteConstraintFunctionBase *function_;
        std::vector<std::size_t> variables_;
    };




    class DiscreteGm
    {

    public:

        using solution_type = std::vector<discrete_label_type>;

        DiscreteGm(const DiscreteSpace & discrete_space);

        template<class NUM_LABELS_ITER>
        DiscreteGm(
            NUM_LABELS_ITER num_labels_begin,
            NUM_LABELS_ITER num_labels_end
        ) :
            space_(num_labels_begin, num_labels_end),
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

        inline DiscreteGm(std::size_t num_var , discrete_label_type num_labels
        ) :
            space_(num_var, num_labels),
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
            return std::max(max_factor_arity_, max_constraint_arity_);
        }

        discrete_label_type num_labels(std::size_t variable_index) const
        {
            return space_[variable_index];
        }
        std::size_t num_variables() const
        {
            return space_.size();
        }
        std::size_t num_factors() const
        {
            return factors_.size();
        }
        std::size_t num_constraints() const
        {
            return constraints_.size();
        }

        template <class F>
        void for_each_factor(F &&f) const
        {
            std::size_t i=0;
            for (const auto &factor : factors_)
            {
                f(factor, i);
                ++i;
            }
        }

        template <class F>
        void for_each_higher_order_factor(F &&f) const
        {
            this->for_each_factor([&](const auto &factor, std::size_t i)
            {
                if (factor.variables().size() > 1)
                {
                    f(factor, i);
                }
            });
        }
        template <class F>
        void for_each_unary_factor(F &&f) const
        {
            this->for_each_factor([&](const auto &factor, std::size_t i)
            {
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





        std::size_t add_energy_function(std::unique_ptr<DiscreteEnergyFunctionBase> function);
        std::size_t add_constraint_function(std::unique_ptr<DiscreteConstraintFunctionBase> function);


        template <class DISCRETE_VARIABLES>
        std::size_t add_factor(
            DISCRETE_VARIABLES &&discrete_variables,
            std::size_t function_id)
        {
            const std::size_t arity = discrete_variables.size();
            max_factor_arity_ = std::max(max_factor_arity_, arity);

            const auto size = shape_product(discrete_variables);
            max_factor_size_ = std::max(max_factor_size_, size);

            factors_.emplace_back(
                std::forward<DISCRETE_VARIABLES>(discrete_variables),
                energy_functions_[function_id].get());
            return factors_.size() - 1;
        }

        template <class DISCRETE_VARIABLES>
        std::size_t add_constraint(
            DISCRETE_VARIABLES &&discrete_variables,
            std::size_t function_id)
        {
            const std::size_t arity = discrete_variables.size();
            max_constraint_arity_ = std::max(max_constraint_arity_, arity);

            const auto size = shape_product(discrete_variables);
            max_constraint_size_ = std::max(max_constraint_size_, size);

            constraints_.emplace_back(
                std::forward<DISCRETE_VARIABLES>(discrete_variables),
                constraint_functions_[function_id].get());
            return constraints_.size() - 1;
        }


        template <class DISCRETE_VARIABLES_INDICES>
        std::size_t add_factor(
            std::initializer_list<DISCRETE_VARIABLES_INDICES> discrete_variables,
            std::size_t function_id)
        {
            const std::size_t arity = discrete_variables.size();
            max_factor_arity_ = std::max(max_factor_arity_, arity);

            const auto size = shape_product(discrete_variables);
            max_factor_size_ = std::max(max_factor_size_, size);

            factors_.emplace_back(
                discrete_variables,
                energy_functions_[function_id].get());
            return factors_.size() - 1;
        }

        template <class DISCRETE_VARIABLES_INDICES>
        std::size_t add_constraint(
            std::initializer_list<DISCRETE_VARIABLES_INDICES> discrete_variables,
            std::size_t function_id)
        {
            const std::size_t arity = discrete_variables.size();
            max_constraint_arity_ = std::max(max_constraint_arity_, arity);
            constraints_.emplace_back(
                discrete_variables,
                constraint_functions_[function_id].get());
            return factors_.size() - 1;
        }

        SolutionValue evaluate(const span<const discrete_label_type> &solution, bool early_stop_infeasible = false) const;
        SolutionValue evaluate(const solution_type &solution, bool early_stop_infeasible = false) const;

        nlohmann::json serialize_json() const;
        static DiscreteGm deserialize_json(const nlohmann::json & json);

    private:

        template <class VARIABLE_INDICIES>
        std::size_t shape_product( VARIABLE_INDICIES && variable_indicies) const
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
        inline DiscreteGmFactorsOfVariables(const DiscreteGm &gm)
        : base_type(gm.space().size())
        {
            for(std::size_t fi=0; fi<gm.factors().size(); ++fi)
            {
                for(const auto &vi : gm.factors()[fi].variables())
                {
                    (*this)[vi].push_back(fi);
                }
            }
        }
    };

    class DiscreteGmConstraintsOfVariables : public std::vector<std::vector<std::size_t>>
    {
    public:
        using base_type = std::vector<std::vector<std::size_t>>;
        inline DiscreteGmConstraintsOfVariables(const DiscreteGm &gm)
        : base_type(gm.space().size())
        {
            for(std::size_t fi=0; fi<gm.constraints().size(); ++fi)
            {
                for(const auto &vi : gm.constraints()[fi].variables())
                {
                    (*this)[vi].push_back(fi);
                }
            }
        }
    };

}
