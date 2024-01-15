#include <limits>
#include <nxtgm/functions/label_count_constraint_base.hpp>
#include <nxtgm/nxtgm.hpp>
#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>

#include <nxtgm/plugins/ilp/ilp_base.hpp>
#include <nxtgm/plugins/plugin_registry.hpp>
#include <nxtgm/utils/timer.hpp>

#include <xtensor/xtensor.hpp>

namespace nxtgm
{

class Acims : public DiscreteGmOptimizerBase
{
    class parameters_type
    {
      public:
        inline parameters_type(OptimizerParameters &&parameters)
        {
            parameters.assign_and_pop("ilp_plugin_name", ilp_plugin_name);
            parameters.assign_and_pop("mode", mode, "chain");
            ensure_all_handled(Acims::name(), parameters);
        }

        std::string ilp_plugin_name;
        std::string mode;
    };

  public:
    using base_type = DiscreteGmOptimizerBase;
    using solution_type = typename DiscreteGm::solution_type;

    using reporter_callback_wrapper_type = typename base_type::reporter_callback_wrapper_type;
    using repair_callback_wrapper_type = typename base_type::repair_callback_wrapper_type;

    inline static std::string name()
    {
        return "Acims";
    }
    virtual ~Acims() = default;

    Acims(const DiscreteGm &gm, OptimizerParameters &&parameters);

    OptimizationStatus optimize_impl(reporter_callback_wrapper_type &, repair_callback_wrapper_type &,
                                     const_discrete_solution_span starting_point) override;

    SolutionValue best_solution_value() const override;
    SolutionValue current_solution_value() const override;

    const solution_type &best_solution() const override;
    const solution_type &current_solution() const override;

  private:
    bool optimize_submodel(const std::vector<uint8_t> &in_model);

    parameters_type parameters_;

    SolutionValue best_solution_value_;
    solution_type best_solution_;
    const LabelCountConstraintBase *constraint_function_;
};

class AcimsFactory : public DiscreteGmOptimizerFactoryBase
{
  public:
    using factory_base_type = DiscreteGmOptimizerFactoryBase;
    virtual ~AcimsFactory() = default;
    expected<std::unique_ptr<DiscreteGmOptimizerBase>> create(const DiscreteGm &gm,
                                                              OptimizerParameters &&params) const override
    {
        return std::make_unique<Acims>(gm, std::move(params));
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
        return "Acims -- Almost conditional independent matching submodel";
    }
};

} // namespace nxtgm

XPLUGIN_CREATE_XPLUGIN_FACTORY(nxtgm::AcimsFactory);

namespace nxtgm
{
Acims::Acims(const DiscreteGm &gm, OptimizerParameters &&parameters)
    : base_type(gm, parameters),
      parameters_(std::move(parameters)),
      best_solution_value_(),
      best_solution_(gm.num_variables(), 0),
      constraint_function_(nullptr)
{
    //
    best_solution_value_ = gm.evaluate(best_solution_, false /* early exit when infeasible*/);

    // TODO -- remove
    // check no constraint besides label count constraint
    if (gm.constraints().size() > 1)
    {
        throw std::runtime_error("Acims only supports label count constraints");
    }

    // TODO -- remove
    // only pairwise factors
    if (gm.max_factor_arity() > 2)
    {
        throw std::runtime_error("Acims only supports pairwise factors");
    }

    // find global matching constraints
    for (const auto &constraint : gm.constraints())
    {
        if (constraint.arity() == gm.num_variables())
        {
            const LabelCountConstraintBase *label_count_constraint =
                dynamic_cast<const LabelCountConstraintBase *>(constraint.function());
            if (label_count_constraint)
            {
                constraint_function_ = label_count_constraint;
                break;
            }
        }
    }
    if (!constraint_function_)
    {
        throw std::runtime_error("no global matching constraint found");
    }
    else
    {
        // ensure min_counts are all 0
        for (std::size_t vi = 0; vi < gm.num_variables(); ++vi)
        {
            if (constraint_function_->min_counts(vi) != 0)
            {
                throw std::runtime_error("min_count must be 0");
            }
        }
    }
}

OptimizationStatus Acims::optimize_impl(reporter_callback_wrapper_type &reporter_callback,
                                        repair_callback_wrapper_type & /*repair_callback not used*/,
                                        const_discrete_solution_span starting_point)
{
    // shortcut to the model
    const auto &gm = this->model();
    const auto num_labels = gm.space().max_num_labels();
    if (starting_point.size() > 0)
    {
        std::copy(starting_point.begin(), starting_point.end(), best_solution_.begin());
    }
    best_solution_value_ = gm.evaluate(best_solution_, false /* early exit when infeasible*/);

    std::vector<uint8_t> in_model(gm.num_variables(), 0);

    ReportData report_data;

    bool changes = true;
    while (changes)
    {
        changes = false;
        if (parameters_.mode == "chain")
        {

            // alternate between even and odd variables
            // been in model and not in model
            for (std::size_t i = 0; i < 2; ++i)
            {
                for (std::size_t vi = 0; vi < gm.num_variables(); ++vi)
                {
                    in_model[vi] = (vi % 2 == i);
                }
                const auto c = optimize_submodel(in_model);
                changes |= c;

                if (!this->report(reporter_callback, report_data))
                {
                    return OptimizationStatus::CALLBACK_EXIT;
                }
            }
        }
        else
        {
            throw std::runtime_error("Unknown mode: " + parameters_.mode);
        }
    }

    return OptimizationStatus::CONVERGED;
}

bool Acims::optimize_submodel(const std::vector<uint8_t> &in_model)
{

    // shortcut to the model
    const auto &gm = this->model();
    const auto num_labels = gm.space().max_num_labels();

    // * map from submodel variable index to variable index
    // * find label usage for variables **not** in the submodel
    std::vector<std::size_t> sub_vi_to_vi;
    std::vector<std::size_t> vi_to_sub_vi(gm.num_variables());
    std::vector<int> label_capacity(num_labels, 0);

    // fill with max_counts from constraint
    for (discrete_label_type l = 0; l < num_labels; ++l)
    {
        label_capacity[l] = constraint_function_->max_counts(l);
    }

    for (std::size_t vi = 0; vi < in_model.size(); ++vi)
    {
        if (in_model[vi])
        {
            vi_to_sub_vi[vi] = sub_vi_to_vi.size();
            sub_vi_to_vi.push_back(vi);
        }
        if (!in_model[vi])
        {
            // reduce label capacity
            label_capacity[best_solution_[vi]]--;
        }
    }
    const auto num_sub_variables = sub_vi_to_vi.size();

    std::vector<std::size_t> labels_to_sub_labels(num_labels);
    std::vector<std::size_t> sub_labels_to_labels;

    // iterate over label_capacity and only keep labels with capacity > 0
    for (std::size_t li = 0; li < label_capacity.size(); ++li)
    {
        if (label_capacity[li] > 0)
        {
            labels_to_sub_labels[li] = sub_labels_to_labels.size();
            sub_labels_to_labels.push_back(li);
        }
    }
    const auto num_sub_labels = sub_labels_to_labels.size();

    // create cost matrix for submodel
    xt::xtensor<double, 2> cost_matrix = xt::zeros<double>({num_sub_variables, num_sub_labels});

    std::vector<energy_type> value_buffer(num_labels);
    std::vector<discrete_label_type> label_buffer(gm.max_factor_arity());

    // iterate all factors of model
    for (const auto &factor : gm.factors())
    {
        const auto &variables = factor.variables();
        const auto arity = factor.arity();

        if (arity == 1 && in_model[variables[0]])
        {
            factor.copy_values(value_buffer.data());
            for (std::size_t li = 0; li < num_labels; ++li)
            {
                if (label_capacity[li] > 0)
                {
                    cost_matrix(vi_to_sub_vi[variables[0]], labels_to_sub_labels[li]) += value_buffer[li];
                }
            }
        }
        else if (factor.arity() == 2)
        {

            // both variables are excluded from submodel
            if (!in_model[variables[0]] && !in_model[variables[1]])
            {
                continue;
            }
            // both variables are included in submodel
            else if (in_model[variables[0]] && in_model[variables[1]])
            {
                throw std::runtime_error(
                    "both variables are included in submodel -- decomposition is wrong -- debug me");
            }
            // first variable is included in submodel
            else if (in_model[variables[0]])
            {
                label_buffer[1] = best_solution_[variables[1]];
                for (std::size_t li = 0; li < num_labels; ++li)
                {
                    if (label_capacity[li] > 0)
                    {
                        label_buffer[0] = li;
                        cost_matrix(vi_to_sub_vi[variables[0]], labels_to_sub_labels[li]) +=
                            factor(label_buffer.data());
                    }
                }
            }
            // second variable is included in submodel
            else if (in_model[variables[1]])
            {
                label_buffer[0] = best_solution_[variables[0]];
                for (std::size_t li = 0; li < num_labels; ++li)
                {
                    if (label_capacity[li] > 0)
                    {
                        label_buffer[1] = li;
                        cost_matrix(vi_to_sub_vi[variables[1]], labels_to_sub_labels[li]) +=
                            factor(label_buffer.data());
                    }
                }
            }
            else
            {
                throw std::runtime_error("one variable is included in submodel -- decomposition is wrong -- debug me");
            }
        }
    }

    // solve the lp
    IlpData ilp_data;
    ilp_data.add_variables(num_sub_variables * num_sub_labels,
                           /*lower bound*/ 0,
                           /*upper bound*/ 1,
                           /*objective*/ 0.0,
                           /*is_integer*/ false);

    // add  marginalization constraints and objective
    std::size_t ilp_var = 0;
    for (std::size_t vi = 0; vi < num_sub_variables; ++vi)
    {
        ilp_data.begin_constraint(1, 1);
        for (std::size_t label = 0; label < num_sub_labels; ++label)
        {
            ilp_data.add_constraint_coefficient(vi * num_sub_labels + label, 1);
            ilp_data[ilp_var] = cost_matrix(vi, label);
            ++ilp_var;
        }
    }

    // add label count constraints
    for (discrete_label_type sub_label = 0; sub_label < num_sub_labels; ++sub_label)
    {
        const auto label = sub_labels_to_labels[sub_label];
        const auto min_count = 0;
        const auto max_count = label_capacity[label];
        ilp_data.begin_constraint(min_count, max_count);
        for (std::size_t vi = 0; vi < num_sub_variables; ++vi)
        {
            ilp_data.add_constraint_coefficient(vi * num_sub_labels + sub_label, 1);
        }
    }

    // solve ILP
    OptimizerParameters parameters;
    parameters["integer"] = false;
    auto factory = get_plugin_registry<IlpFactoryBase>().get_factory(std::string("ilp_") + parameters_.ilp_plugin_name);
    auto ilp_solver = factory->create(std::move(ilp_data), std::move(parameters));
    ilp_solver->optimize(nullptr);
    std::vector<double> solution(num_sub_variables * num_sub_labels);
    ilp_solver->get_solution(solution.data());

    bool any_change = false;
    for (std::size_t subvi = 0; subvi < num_sub_variables; ++subvi)
    {
        for (std::size_t sublabel = 0; sublabel < num_sub_labels; ++sublabel)
        {
            const auto lp_sol = solution[subvi * num_sub_labels + sublabel];
            if (lp_sol >= 0.9)
            {
                const auto old_label = best_solution_[sub_vi_to_vi[subvi]];
                const auto new_label = sub_labels_to_labels[sublabel];
                if (old_label != new_label)
                {
                    any_change = true;
                }
                best_solution_[sub_vi_to_vi[subvi]] = new_label;
                break;
            }
        }
    }
    if (any_change)
    {
        best_solution_value_ = gm.evaluate(best_solution_, false /* early exit when infeasible*/);
    }
    return any_change;
}

SolutionValue Acims::best_solution_value() const
{
    return best_solution_value_;
}
SolutionValue Acims::current_solution_value() const
{
    return best_solution_value_;
}

const typename Acims::solution_type &Acims::best_solution() const
{
    return best_solution_;
}
const typename Acims::solution_type &Acims::current_solution() const
{
    return best_solution_;
}

} // namespace nxtgm
