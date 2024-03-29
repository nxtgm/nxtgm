#include <limits>
#include <nxtgm/nxtgm.hpp>
#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>
#include <nxtgm/plugins/min_st_cut/min_st_cut_base.hpp>
#include <nxtgm/plugins/plugin_registry.hpp>
#include <nxtgm/utils/timer.hpp>

namespace nxtgm
{

class GraphCut : public DiscreteGmOptimizerBase
{
    class parameters_type
    {
      public:
        inline parameters_type(OptimizerParameters &&parameters)
        {
            parameters.assign_and_pop("min_st_cut_plugin_name", min_st_cut_plugin_name);
            parameters.assign_and_pop("submodular_epsilon", submodular_epsilon);
            parameters.assign_and_pop("constraint_scaling", constraint_scaling);
            ensure_all_handled(GraphCut::name(), parameters);
        }

        std::string min_st_cut_plugin_name;
        double submodular_epsilon = 1e-6;
        energy_type constraint_scaling = default_constraint_scaling;
    };

  public:
    using base_type = DiscreteGmOptimizerBase;
    using solution_type = typename DiscreteGm::solution_type;

    using reporter_callback_wrapper_type = typename base_type::reporter_callback_wrapper_type;
    using repair_callback_wrapper_type = typename base_type::repair_callback_wrapper_type;

    inline static std::string name()
    {
        return "GraphCut";
    }
    virtual ~GraphCut() = default;

    GraphCut(const DiscreteGm &gm, OptimizerParameters &&parameters);

    OptimizationStatus optimize_impl(reporter_callback_wrapper_type &, repair_callback_wrapper_type &,
                                     const_discrete_solution_span starting_point) override;

    SolutionValue best_solution_value() const override;
    SolutionValue current_solution_value() const override;

    const solution_type &best_solution() const override;
    const solution_type &current_solution() const override;

  private:
    parameters_type parameters_;

    SolutionValue best_solution_value_;
    solution_type best_solution_;

    std::unique_ptr<MinStCutBase> min_st_cut_;
};

class GraphCutFactory : public DiscreteGmOptimizerFactoryBase
{
  public:
    using factory_base_type = DiscreteGmOptimizerFactoryBase;
    virtual ~GraphCutFactory() = default;
    expected<std::unique_ptr<DiscreteGmOptimizerBase>> create(const DiscreteGm &gm,
                                                              OptimizerParameters &&params) const override
    {
        return std::make_unique<GraphCut>(gm, std::move(params));
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
        return "Graph cut optimizer";
    }
};

} // namespace nxtgm

XPLUGIN_CREATE_XPLUGIN_FACTORY(nxtgm::GraphCutFactory);

namespace nxtgm
{
GraphCut::GraphCut(const DiscreteGm &gm, OptimizerParameters &&parameters)
    : base_type(gm, parameters),
      parameters_(std::move(parameters)),
      best_solution_value_(),
      best_solution_(gm.num_variables(), 0),
      min_st_cut_(nullptr)
{

    best_solution_value_ = gm.evaluate(best_solution_, false /* early exit when infeasible*/);

    // check space
    if (gm.space().max_num_labels() != 2)
    {
        throw UnsupportedModelException("GraphCut only supports binary variables");
    }
    // check arity
    if (gm.max_arity() > 2)
    {
        throw UnsupportedModelException("GraphCut only supports pairwise factors");
    }

    size_t num_edges = 0;
    for (const auto &factor : gm.factors())
    {
        if (const auto arity = factor.arity(); arity == 2)
            num_edges += 1;
    }
    auto factory = get_plugin_registry<MinStCutFactoryBase>().get_factory(parameters_.min_st_cut_plugin_name);

    min_st_cut_ = factory->create(gm.num_variables(), num_edges);

    double energies[4] = {0, 0, 0, 0};

    gm.for_each_factor_and_constraint(
        [&](auto &&factor_or_constraint, std::size_t /*factor_or_constraint_index*/, bool is_constraint) {
            if (const auto arity = factor_or_constraint.arity(); arity == 1)
            {
                const auto var0 = factor_or_constraint.variable(0);
                factor_or_constraint.function()->copy_values(energies);
                if (energies[0] <= energies[1])
                {
                    const auto value = energies[1] - energies[0];
                    const auto c = is_constraint ? value * parameters_.constraint_scaling : value;
                    min_st_cut_->add_terminal_weights(var0, c, 0);
                }
                else
                {
                    const auto value = energies[0] - energies[1];
                    const auto c = is_constraint ? value * parameters_.constraint_scaling : value;
                    min_st_cut_->add_terminal_weights(var0, 0, c);
                }
            }
            else if (arity == 2)
            {
                const auto scale = is_constraint ? parameters_.constraint_scaling : 1.0;
                factor_or_constraint.function()->copy_values(energies);
                const auto A = energies[0] * scale; // 00
                const auto B = energies[1] * scale; // 01
                const auto C = energies[2] * scale; // 10
                const auto D = energies[3] * scale; // 11
                const auto var0 = factor_or_constraint.variable(0);
                const auto var1 = factor_or_constraint.variable(1);

                // first variable
                if (C > A)
                {
                    min_st_cut_->add_terminal_weights(var0, C - A, 0);
                }
                else if (C < A)
                {
                    min_st_cut_->add_terminal_weights(var0, 0, A - C);
                }

                // second variable
                if (D > C)
                {
                    min_st_cut_->add_terminal_weights(var1, D - C, 0);
                }
                else if (D < C)
                {
                    min_st_cut_->add_terminal_weights(var1, 0, C - D);
                }

                // submodular term
                const auto term = B + C - A - D;
                if (term < -parameters_.submodular_epsilon)
                {
                    throw std::runtime_error("non sub-modular factors cannot be processed");
                }
                if (term > 0)
                {
                    min_st_cut_->add_edge(var0, var1, term, 0);
                }
            }
        });
}

OptimizationStatus GraphCut::optimize_impl(reporter_callback_wrapper_type &reporter_callback,
                                           repair_callback_wrapper_type & /*repair_callback not used*/,
                                           const_discrete_solution_span)
{
    // shortcut to the model
    const auto &gm = this->model();

    /*const auto flow = */ min_st_cut_->solve(best_solution_.data());
    best_solution_value_ = gm.evaluate(best_solution_, false /* early exit when infeasible*/);

    return OptimizationStatus::OPTIMAL;
}

SolutionValue GraphCut::best_solution_value() const
{
    return best_solution_value_;
}
SolutionValue GraphCut::current_solution_value() const
{
    return best_solution_value_;
}

const typename GraphCut::solution_type &GraphCut::best_solution() const
{
    return best_solution_;
}
const typename GraphCut::solution_type &GraphCut::current_solution() const
{
    return best_solution_;
}

} // namespace nxtgm
