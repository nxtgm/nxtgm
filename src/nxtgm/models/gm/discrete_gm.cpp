#include <nxtgm/models/gm/discrete_gm.hpp>

#include <nxtgm/nxtgm.hpp>
#include <nxtgm/models/gm/discrete_gm.hpp>

namespace nxtgm
{
    const_discrete_label_span local_solution_from_model_solution(
        const std::vector<std::size_t> & variables,
        const std::vector<discrete_label_type> & solution,        
        std::vector<discrete_label_type> & local_labels_buffer
    ){
        const auto arity = variables.size();
        if(local_labels_buffer.size() < arity){
            local_labels_buffer.resize(arity*2);
        }

        auto i=0;
        for(auto vi : variables){
            local_labels_buffer[i] = solution[vi];
            i++;
        }
        return const_discrete_label_span(local_labels_buffer.data(), arity);
        
    }

    DiscreteGm::DiscreteGm(const DiscreteSpace & discrete_space)
    :   discrete_space_(discrete_space),
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


    std::size_t DiscreteGm::add_energy_function(std::unique_ptr<DiscreteEnergyFunctionBase> function)
    {
        energy_functions_.push_back(std::move(function));
        return energy_functions_.size() - 1;
    }
    
    std::size_t DiscreteGm::add_constraint_function(std::unique_ptr<DiscreteConstraintFunctionBase> function)
    {
        constraint_functions_.push_back(std::move(function));
        return constraint_functions_.size() - 1;
    }

    SolutionValue DiscreteGm::operator()(const solution_type &solution, bool early_stop_infeasible) const
    {
        bool total_is_feasible = true;
        energy_type total_how_violated = 0;
        
        // buffer holding the labels for the factors/constraints
        std::vector<discrete_label_type> local_labels_buffer(2);

        for(const auto & constraint : constraints_)
        {
            const const_discrete_label_span labels = local_solution_from_model_solution(
                constraint.variables(),
                solution,
                local_labels_buffer
            );
            const auto [is_feasible, how_violated] = constraint.function()->feasible(labels);
            if(!is_feasible)
            {
                if(early_stop_infeasible)
                {
                    return SolutionValue{std::numeric_limits<energy_type>::infinity(), false, how_violated};
                }
                else
                {
                    total_is_feasible = false;
                    total_how_violated += how_violated;
                }
            }
            else{
                total_how_violated += how_violated;
            }
        }

        energy_type total_energy = 0;
        for (const auto &factor : factors_)
        {   

            const const_discrete_label_span labels = local_solution_from_model_solution(
                factor.variables(),
                solution,
                local_labels_buffer
            );
            total_energy += factor.function()->energy(labels);
        }
        return SolutionValue{total_energy, total_is_feasible, total_how_violated};
    }
} 
