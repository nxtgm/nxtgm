#include <chrono>
#include <nxtgm/models/gm/discrete_gm.hpp>
#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>

#include <nxtgm/utils/n_nested_loops.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>

#include <nxtgm/plugins/ilp/ilp_base.hpp>
#include <nxtgm/plugins/plugin_registry.hpp>

namespace nxtgm
{

class IlpBased : public DiscreteGmOptimizerBase
{
    class parameters_type
    {
      public:
        inline parameters_type(OptimizerParameters &parameters)
        {
            parameters.assign_and_pop("integer", integer);
            parameters.assign_and_pop("ilp_plugin_name", ilp_plugin_name);
            parameters.assign_and_pop("ilp_plugin_parameters", ilp_plugin_parameters);
        }

        bool integer = true;
        std::string ilp_plugin_name = "highs";
        OptimizerParameters ilp_plugin_parameters;
    };

  public:
    using base_type = DiscreteGmOptimizerBase;
    using solution_type = typename DiscreteGm::solution_type;

    using reporter_callback_wrapper_type = typename base_type::reporter_callback_wrapper_type;
    using repair_callback_wrapper_type = typename base_type::repair_callback_wrapper_type;

    inline static std::string name()
    {
        return "IlpBased";
    }

    IlpBased(const DiscreteGm &gm, OptimizerParameters &&parameters);

    virtual ~IlpBased() = default;

    OptimizationStatus optimize_impl(reporter_callback_wrapper_type &reporter_callback,
                                     repair_callback_wrapper_type & /*repair_callback not used*/,
                                     const_discrete_solution_span starting_point) override;

    SolutionValue best_solution_value() const override;
    SolutionValue current_solution_value() const override;

    const solution_type &best_solution() const override;
    const solution_type &current_solution() const override;

    energy_type lower_bound() const override;

  private:
    void setup_ilp();

    parameters_type parameters_;

    solution_type best_solution_;
    solution_type current_solution_;
    SolutionValue best_sol_value_;
    SolutionValue current_sol_value_;

    energy_type lower_bound_;

    // map from variable index to the beginning of the indicator variables
    IndicatorVariableMapping indicator_variable_mapping_;

    std::unique_ptr<IlpBase> ilp_;
};

class IlpBasedFactory : public DiscreteGmOptimizerFactoryBase
{
  public:
    using factory_base_type = DiscreteGmOptimizerFactoryBase;
    virtual ~IlpBasedFactory() = default;
    expected<std::unique_ptr<DiscreteGmOptimizerBase>> create(const DiscreteGm &gm,
                                                              OptimizerParameters &&params) const override
    {
        return std::make_unique<IlpBased>(gm, std::move(params));
    }
    int priority() const override
    {
        return plugin_priority(PluginPriority::MEDIUM);
    }
    std::string license() const override
    {
        return "MIT";
    }
    std::string description() const override
    {
        return "Iterated conditional models optimizer";
    }
    OptimizerFlags flags() const override
    {
        return OptimizerFlags::Optimal;
    }
};

} // namespace nxtgm

XPLUGIN_CREATE_XPLUGIN_FACTORY(nxtgm::IlpBasedFactory);

namespace nxtgm
{

IlpBased::IlpBased(const DiscreteGm &gm, OptimizerParameters &&parameters)
    : base_type(gm, parameters),
      parameters_(parameters),
      best_solution_(gm.num_variables(), 0),
      current_solution_(),
      best_sol_value_(),
      current_sol_value_(),
      lower_bound_(-std::numeric_limits<energy_type>::infinity()),
      indicator_variable_mapping_(gm.space()),
      ilp_(nullptr)
{
    ensure_all_handled(name(), parameters);
    best_sol_value_ = this->model().evaluate(best_solution_, false);
    current_solution_ = best_solution_;
    current_sol_value_ = best_sol_value_;

    // std::cout<<"setup_ilp"<<std::endl;
    this->setup_ilp();
    // std::cout<<"setup_ilp done"<<std::endl;
}

void IlpBased::setup_ilp()
{
    // shortcuts
    const auto &model = this->model();
    const auto &space = model.space();

    IlpData ilp_data;

    // add inter variables for all the indicator variables
    // (objective will be added later)
    ilp_data.add_variables(indicator_variable_mapping_.num_indicator_variables(), 0.0, 1.0, 0.0, true);

    // sum to one constraints
    for (std::size_t vi = 0; vi < space.size(); ++vi)
    {
        ilp_data.begin_constraint(1.0, 1.0);
        for (discrete_label_type l = 0; l < space[vi]; ++l)
        {
            ilp_data.add_constraint_coefficient(indicator_variable_mapping_[vi] + l, 1.0);
        }
    }

    // add all the factors to the ilp
    std::vector<std::size_t> indicator_variables_mapping_buffer(model.max_arity());

    for (auto &&factor : model.factors())
    {
        factor.map_from_model(indicator_variable_mapping_, indicator_variables_mapping_buffer);
        factor.function()->add_to_lp(ilp_data, indicator_variables_mapping_buffer.data());
    };

    // add constraints to the ilp
    for (auto &&constraint : model.constraints())
    {
        constraint.map_from_model(indicator_variable_mapping_, indicator_variables_mapping_buffer);
        constraint.function()->add_to_lp(ilp_data, indicator_variables_mapping_buffer.data());
    }

    auto factory = get_plugin_registry<IlpFactoryBase>().get_factory(std::string("ilp_") + parameters_.ilp_plugin_name);
    parameters_.ilp_plugin_parameters["integer"] = parameters_.integer;
    ilp_ = factory->create(std::move(ilp_data), std::move(parameters_.ilp_plugin_parameters));
}

OptimizationStatus IlpBased::optimize_impl(reporter_callback_wrapper_type &reporter_callback,
                                           repair_callback_wrapper_type & /*repair_callback not used*/,
                                           const_discrete_solution_span starting_point)
{
    OptimizationStatus status = OptimizationStatus::OPTIMAL;

    if (starting_point.size() > 0)
    {
        // optimize **with** starting point
        std::copy(starting_point.begin(), starting_point.end(), best_solution_.begin());
        best_sol_value_ = this->model().evaluate(best_solution_, false);
        current_solution_ = best_solution_;
        current_sol_value_ = best_sol_value_;

        // map the starting point to the lp starting point
        std::vector<double> lp_starting_point(ilp_->num_variables(), 0.0);
        for (std::size_t i = 0; i < best_solution_.size(); ++i)
        {
            lp_starting_point[indicator_variable_mapping_[i] + best_solution_[i]] = 1.0;
        }
        status = ilp_->optimize(lp_starting_point.data());
    }

    else
    {
        // optimize **without** starting point
        status = ilp_->optimize(nullptr);
    }

    if (status == OptimizationStatus::INFEASIBLE || status == OptimizationStatus::TIME_LIMIT_REACHED)
    {
        return status;
    }

    // get the solution
    std::vector<double> solution(ilp_->num_variables());
    ilp_->get_solution(solution.data());

    const bool all_integral =
        indicator_variable_mapping_.lp_solution_to_model_solution(solution.data(), best_solution_);

    this->best_sol_value_ = this->model().evaluate(this->best_solution_);

    return status;
}

energy_type IlpBased::lower_bound() const
{
    return this->lower_bound_;
}

SolutionValue IlpBased::best_solution_value() const
{
    return this->best_sol_value_;
}
SolutionValue IlpBased::current_solution_value() const
{
    return this->current_sol_value_;
}

const typename IlpBased::solution_type &IlpBased::best_solution() const
{
    return this->best_solution_;
}
const typename IlpBased::solution_type &IlpBased::current_solution() const
{
    return this->current_solution_;
}

} // namespace nxtgm
