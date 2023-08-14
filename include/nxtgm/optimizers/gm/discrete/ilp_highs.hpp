#pragma once
#include <chrono>
#include <highs/Highs.h>
#include <nxtgm/models/gm/discrete_gm.hpp>
#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>

namespace nxtgm
{

class IlpHighs : public DiscreteGmOptimizerBase
{
  public:
    class parameters_type : public OptimizerParametersBase
    {
      public:
        inline parameters_type(const nlohmann::json &json_parameters)
            : OptimizerParametersBase(json_parameters)
        {
            if (json_parameters.contains("integer"))
            {
                integer = json_parameters["integer"];
            }
            if (json_parameters.contains("highs_log_to_console"))
            {
                highs_log_to_console = json_parameters["highs_log_to_console"];
            }
        }

        bool integer = true;
        bool highs_log_to_console = false;
    };

    using base_type = DiscreteGmOptimizerBase;
    using solution_type = typename DiscreteGm::solution_type;

    using reporter_callback_wrapper_type = typename base_type::reporter_callback_wrapper_type;
    using repair_callback_wrapper_type = typename base_type::repair_callback_wrapper_type;

    using base_type::optimize;

    inline static std::string name()
    {
        return "IlpHighs";
    }

    IlpHighs(const DiscreteGm &gm, const parameters_type &parameters,
             const solution_type &initial_solution = solution_type());

    virtual ~IlpHighs() = default;

    OptimizationStatus optimize(reporter_callback_wrapper_type &reporter_callback,
                                repair_callback_wrapper_type & /*repair_callback not used*/,
                                const_discrete_solution_span starting_point) override;

    SolutionValue best_solution_value() const override;
    SolutionValue current_solution_value() const override;

    const solution_type &best_solution() const override;
    const solution_type &current_solution() const override;

    energy_type lower_bound() const override;

  private:
    void setup_lp();

    parameters_type parameters_;

    solution_type best_solution_;
    solution_type current_solution_;
    SolutionValue best_sol_value_;
    SolutionValue current_sol_value_;

    energy_type lower_bound_;

    // map from variable index to the beginning of the indicator variables
    IndicatorVariableMapping indicator_variable_mapping_;

    IlpData ilp_data_;

    HighsModel highs_model_;
};
} // namespace nxtgm
