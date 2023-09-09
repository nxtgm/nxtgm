
#include <nxtgm/nxtgm.hpp>
#include <nxtgm/optimizers/gm/discrete/discrete_gm_optimizer_factory.hpp>
#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>

namespace nxtgm
{

class ReducedGmOptimizer : public DiscreteGmOptimizerBase
{
    class parameters_type
    {
      public:
        inline parameters_type(OptimizerParameters &parameters)
        {
            if (auto find = parameters.string_parameters.find("sub_optimizer");
                find != parameters.string_parameters.end())
            {
                sub_optimizer = find->second;
                parameters.string_parameters.erase(find);
            }
            if (auto find = parameters.optimizer_parameters.find("sub_optimizer_parameters");
                find != parameters.optimizer_parameters.end())
            {
                sub_optimizer_parameters = find->second;
                parameters.optimizer_parameters.erase(find);
            }
        }

        std::string sub_optimizer = "icm";
        OptimizerParameters sub_optimizer_parameters = OptimizerParameters();
    };

  public:
    using base_type = DiscreteGmOptimizerBase;
    using solution_type = typename DiscreteGm::solution_type;

    using reporter_callback_wrapper_type = typename base_type::reporter_callback_wrapper_type;
    using repair_callback_wrapper_type = typename base_type::repair_callback_wrapper_type;

    inline static std::string name()
    {
        return "ReducedGmOptimizer";
    }
    virtual ~ReducedGmOptimizer() = default;

    ReducedGmOptimizer(const DiscreteGm &gm, OptimizerParameters &&parameters);

    OptimizationStatus optimize_impl(reporter_callback_wrapper_type &, repair_callback_wrapper_type &,
                                     const_discrete_solution_span starting_point) override;

    SolutionValue best_solution_value() const override;
    SolutionValue current_solution_value() const override;

    const solution_type &best_solution() const override;
    const solution_type &current_solution() const override;

  private:
    OptimizationStatus get_partial_optimality(reporter_callback_wrapper_type &, repair_callback_wrapper_type &);
    OptimizationStatus get_partial_optimality_via_qpbo(reporter_callback_wrapper_type &,
                                                       repair_callback_wrapper_type &);

    OptimizationStatus build_and_optimize_submodel(reporter_callback_wrapper_type &, repair_callback_wrapper_type &,
                                                   const_discrete_solution_span);

    void compute_labels();

    parameters_type parameters_;
    SolutionValue best_solution_value_;
    solution_type best_solution_;
    energy_type lower_bound_;

    std::vector<uint8_t> is_partial_optimal_;
    std::size_t n_partial_optimal_;
};

class ReducedGmOptimizerFactory : public DiscreteGmOptimizerFactoryBase
{
  public:
    using factory_base_type = DiscreteGmOptimizerFactoryBase;
    virtual ~ReducedGmOptimizerFactory() = default;
    std::unique_ptr<DiscreteGmOptimizerBase> create(const DiscreteGm &gm, OptimizerParameters &&params) const override
    {
        return std::make_unique<ReducedGmOptimizer>(gm, std::move(params));
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
        return "Find partial optimal solution and then optimize the reduced model.";
    }
    OptimizerFlags flags() const override
    {
        return OptimizerFlags::PartialOptimal | OptimizerFlags::OptimalOnBinarySecondOrderSubmodular |
               OptimizerFlags::OptimalOnTrees | OptimizerFlags::WarmStartable | OptimizerFlags::MetaOptimizer;
    }
};

} // namespace nxtgm

XPLUGIN_CREATE_XPLUGIN_FACTORY(nxtgm::ReducedGmOptimizerFactory);

namespace nxtgm
{

ReducedGmOptimizer::ReducedGmOptimizer(const DiscreteGm &gm, OptimizerParameters &&parameters)
    : base_type(gm, parameters),
      parameters_(parameters),
      best_solution_value_(),
      best_solution_(gm.num_variables(), 0),
      lower_bound_(-std::numeric_limits<energy_type>::infinity()),
      is_partial_optimal_(gm.num_variables(), 0),
      n_partial_optimal_(0)
{
    ensure_all_handled(name(), parameters);
    best_solution_value_ = this->model().evaluate(best_solution_);
}

OptimizationStatus ReducedGmOptimizer::optimize_impl(reporter_callback_wrapper_type &reporter_callback,
                                                     repair_callback_wrapper_type &repair_callback,
                                                     const_discrete_solution_span starting_point)
{
    // get the partial optimal variables
    OptimizationStatus status = this->get_partial_optimality(reporter_callback, repair_callback);
    if (status == OptimizationStatus::OPTIMAL || status == OptimizationStatus::INFEASIBLE)
    {
        return status;
    }

    // build and optimize the submodel (and map to the original model)
    return this->build_and_optimize_submodel(reporter_callback, repair_callback, starting_point);
}

SolutionValue ReducedGmOptimizer::best_solution_value() const
{
    return best_solution_value_;
}
SolutionValue ReducedGmOptimizer::current_solution_value() const
{
    return best_solution_value_;
}

const typename ReducedGmOptimizer::solution_type &ReducedGmOptimizer::best_solution() const
{
    return best_solution_;
}
const typename ReducedGmOptimizer::solution_type &ReducedGmOptimizer::current_solution() const
{
    return best_solution_;
}

OptimizationStatus ReducedGmOptimizer::get_partial_optimality(reporter_callback_wrapper_type &reporter_callback,
                                                              repair_callback_wrapper_type &repair_callback)
{
    return this->get_partial_optimality_via_qpbo(reporter_callback, repair_callback);
}

OptimizationStatus ReducedGmOptimizer::get_partial_optimality_via_qpbo(
    reporter_callback_wrapper_type & /*reporter_callback*/, repair_callback_wrapper_type & /*repair_callback*/
)
{
    OptimizerParameters qpbo_parameters;
    qpbo_parameters["strong_persistencies"] = false;

    const auto qpbo_type = this->model().max_arity() > 2 ? std::string("hqpbo") : std::string("qpbo");

    auto optimizer = discrete_gm_optimizer_factory(this->model(), qpbo_type, qpbo_parameters);
    optimizer->optimize();
    best_solution_ = optimizer->best_solution();
    best_solution_value_ = optimizer->best_solution_value();
    lower_bound_ = optimizer->lower_bound();

    n_partial_optimal_ = 0;
    for (std::size_t i = 0; i < this->model().num_variables(); ++i)
    {
        is_partial_optimal_[i] = optimizer->is_partial_optimal(i);
        if (is_partial_optimal_[i])
        {
            ++n_partial_optimal_;
        }
    }
    if (n_partial_optimal_ == this->model().num_variables())
    {
        return OptimizationStatus::OPTIMAL;
    }
    else if (n_partial_optimal_ > 0)
    {
        return OptimizationStatus::PARTIAL_OPTIMAL;
    }
    else /*if( n_partial_optimal_ == 0)*/
    {
        return OptimizationStatus::CONVERGED;
    }
}

// helper function to use strictest time limit

OptimizationStatus ReducedGmOptimizer::build_and_optimize_submodel(reporter_callback_wrapper_type &,
                                                                   repair_callback_wrapper_type &,
                                                                   const_discrete_solution_span starting_point)
{
    const auto &gm = this->model();

    if (n_partial_optimal_ == 0)
    {
        auto factory = get_discrete_gm_optimizer_factory(parameters_.sub_optimizer);

        auto optimizer = factory->create(gm, OptimizerParameters(parameters_.sub_optimizer_parameters));
        if (this->remaining_time() < optimizer->get_time_limit())
        {
            optimizer->set_time_limit(this->remaining_time());
        }

        auto status = optimizer->optimize(nullptr, nullptr, starting_point);

        if (auto solution_value = optimizer->best_solution_value(); solution_value < best_solution_value_)
        {
            best_solution_value_ = solution_value;
            best_solution_ = optimizer->best_solution();
        }
        if (auto lb = optimizer->lower_bound(); lb > lower_bound_)
        {
            lower_bound_ = lb;
        }
        return status;
    }
    else
    {
        auto mask = span<const uint8_t>(is_partial_optimal_.data(), is_partial_optimal_.size());
        auto labels = span<const discrete_label_type>(best_solution_.data(), best_solution_.size());

        auto [sub_gm, gm_to_sub_gm, constant] = gm.bind(mask, labels, false);

        auto sub_optimizer =
            discrete_gm_optimizer_factory(sub_gm, parameters_.sub_optimizer, parameters_.sub_optimizer_parameters);

        OptimizationStatus sub_status;
        if (starting_point.empty())
        {
            sub_status = sub_optimizer->optimize();
        }
        else
        {
            // build starting point for submodel
            std::vector<discrete_label_type> sub_starting_point(sub_gm.num_variables());
            for (auto [gm_vi, sub_gm_vi] : gm_to_sub_gm)
            {
                sub_starting_point[sub_gm_vi] = starting_point[gm_vi];
            }
            const const_discrete_label_span sub_start_point_span(sub_starting_point.data(), sub_starting_point.size());
            sub_status = sub_optimizer->optimize(nullptr, nullptr, sub_start_point_span);
        }

        if (sub_status == OptimizationStatus::INFEASIBLE)
        {
            return sub_status;
        }

        const auto &sub_best_solution_value = sub_optimizer->best_solution_value();
        const auto &sub_best_solution = sub_optimizer->best_solution();

        const auto solution_value = sub_best_solution_value + constant;
        if (solution_value < best_solution_value_)
        {
            best_solution_value_ = solution_value + constant;
            for (auto [gm_vi, sub_gm_vi] : gm_to_sub_gm)
            {
                best_solution_[gm_vi] = sub_best_solution[sub_gm_vi];
            }
        }
        return sub_status;
    }
}

} // namespace nxtgm
