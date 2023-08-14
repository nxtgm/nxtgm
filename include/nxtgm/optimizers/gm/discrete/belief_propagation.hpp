#pragma once

#include <chrono>
#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>

namespace nxtgm
{

class BeliefPropagation : public DiscreteGmOptimizerBase
{
    class parameters_type : public OptimizerParametersBase
    {
      public:
        inline parameters_type(const nlohmann::json &json_parameters)
            : OptimizerParametersBase(json_parameters)
        {
            if (json_parameters.contains("max_iterations"))
            {
                max_iterations = json_parameters["max_iterations"];
            }
            if (json_parameters.contains("convergence_tolerance"))
            {
                convergence_tolerance = json_parameters["convergence_tolerance"];
            }
            if (json_parameters.contains("damping"))
            {
                damping = json_parameters["damping"];
            }
            if (json_parameters.contains("normalize_messages"))
            {
                normalize_messages = json_parameters["normalize_messages"];
            }
        }

        std::size_t max_iterations = 1000;
        energy_type convergence_tolerance = 1e-5;
        energy_type damping = 0.0;
        bool normalize_messages = true;
    };

  public:
    using base_type = DiscreteGmOptimizerBase;
    using solution_type = typename DiscreteGm::solution_type;

    using reporter_callback_wrapper_type = typename base_type::reporter_callback_wrapper_type;
    using repair_callback_wrapper_type = typename base_type::repair_callback_wrapper_type;

    using base_type::optimize;

    inline static std::string name()
    {
        return "BeliefPropagation";
    }
    virtual ~BeliefPropagation() = default;

    BeliefPropagation(const DiscreteGm &gm, const nlohmann::json &json_parameters)

        OptimizationStatus optimize(reporter_callback_wrapper_type &, repair_callback_wrapper_type &,
                                    const_discrete_solution_span starting_point) override;

    SolutionValue best_solution_value() const override;
    SolutionValue current_solution_value() const override;

    const solution_type &best_solution() const override;
    const solution_type &current_solution() const override;

  private:
    void compute_variable_to_factor_messages();
    void compute_factor_to_variable_messages();
    void compute_beliefs();
    void compute_solution();

    energy_type compute_convergence_delta();
    void damp_messages();

    parameters_type parameters_;
    std::size_t iteration_;

    // since we do damping, we need to store old and current messages
    std::vector<energy_type> message_storage_[2];

    // old and new beliefs
    std::vector<energy_type> belief_storage_;

    // offsets for the messages
    std::vector<std::size_t> factor_to_variable_message_offsets_;
    std::vector<std::size_t> variable_to_factor_message_offsets_;
    std::vector<std::size_t> belief_offsets_;

    std::vector<energy_type *> local_factor_to_variable_messages_;
    std::vector<energy_type *> local_variable_to_factor_messages_;

    std::vector<discrete_label_type> max_arity_label_buffer;

    SolutionValue best_solution_value_;
    SolutionValue current_solution_value_;
    solution_type best_solution_;
    solution_type current_solution_;
};
} // namespace nxtgm
