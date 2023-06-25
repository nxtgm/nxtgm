#pragma once

#include <nxtgm/energy_functions/discrete_energy_function_base.hpp>
#include <nxtgm/constraint_functions/discrete_constraint_function_base.hpp>
#include <nxtgm/models/solution_value.hpp>
#include <nxtgm/spaces/discrete_space.hpp>

#include <nxtgm/nxtgm.hpp>

#include <vector>
#include <iostream>
#include <numeric>

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


        template<class MODEL_DATA, class CONSTRAINT_DATA>
        void map_from_model(const MODEL_DATA & model_data, CONSTRAINT_DATA & constraint_data)const
        {
            const auto & variables = this->variables();
            for(std::size_t ai = 0; ai < variables.size(); ++ai)
            {
                constraint_data[ai] = model_data[variables[ai]];
            }
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

        inline const DiscreteSpace &space() const
        {
            return discrete_space_;
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


        SolutionValue operator()(const solution_type &solution, bool early_stop_infeasible) const;

    private:

        template <class VARIABLE_INDICIES>
        std::size_t shape_product( VARIABLE_INDICIES && variable_indicies) const
        {
            std::size_t product = 1;
            for (const auto &variable_index : variable_indicies)
            {
                product *= discrete_space_[variable_index];
            }
            return product;
        }

    
        DiscreteSpace discrete_space_;
        std::vector<DiscreteFactor> factors_;
        std::vector<std::unique_ptr<DiscreteEnergyFunctionBase>> energy_functions_;

        std::vector<DiscreteConstraint> constraints_;
        std::vector<std::unique_ptr<DiscreteConstraintFunctionBase>> constraint_functions_;

        std::size_t max_factor_arity_;
        std::size_t max_constraint_arity_;

        std::size_t max_factor_size_;
        std::size_t max_constraint_size_;
    };

} 
