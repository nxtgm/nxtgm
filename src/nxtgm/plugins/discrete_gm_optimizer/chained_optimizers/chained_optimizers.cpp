#include <nxtgm/nxtgm.hpp>
#include <nxtgm/optimizers/gm/discrete/discrete_gm_optimizer_factory.hpp>
#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>
#include <nxtgm/utils/timer.hpp>

namespace nxtgm
{

class ChainedOptimizers : public DiscreteGmOptimizerBase
{

    class parameters_type
    {
      public:
        inline parameters_type(OptimizerParameters &parameters)
        {
            std::cout << "get optimizer names" << std::endl;
            parameters.assign_and_pop("optimizer_names", optimizer_names);

            std::cout << "get optimizer parameters" << std::endl;
            parameters.assign_and_pop("optimizer_parameters", optimizer_parameters);

            if (optimizer_names.size() != optimizer_parameters.size())
            {
                throw std::runtime_error("optimizer_names and optimizer_parameters must have the same size");
            }
        }

        std::vector<std::string> optimizer_names;
        std::vector<OptimizerParameters> optimizer_parameters;
    };

  public:
    using base_type = DiscreteGmOptimizerBase;
    using solution_type = typename DiscreteGm::solution_type;

    using reporter_callback_wrapper_type = typename base_type::reporter_callback_wrapper_type;
    using repair_callback_wrapper_type = typename base_type::repair_callback_wrapper_type;

    inline static std::string name()
    {
        return "ChainedOptimizers";
    }
    virtual ~ChainedOptimizers() = default;

    ChainedOptimizers(const DiscreteGm &gm, OptimizerParameters &&parameters);

    OptimizationStatus optimize_impl(reporter_callback_wrapper_type &, repair_callback_wrapper_type &,
                                     const_discrete_solution_span starting_point) override;

    SolutionValue best_solution_value() const override;
    SolutionValue current_solution_value() const override;

    const solution_type &best_solution() const override;
    const solution_type &current_solution() const override;

    bool is_partial_optimal(std::size_t variable_index) const override;

  private:
    parameters_type parameters_;

    SolutionValue best_solution_value_;
    solution_type best_solution_;
    std::vector<uint8_t> is_partial_optimal_;

    DiscreteGmOptimizerBase *_current_optimizer;
};

class ChainedOptimizerFactory : public DiscreteGmOptimizerFactoryBase
{
  public:
    using factory_base_type = DiscreteGmOptimizerFactoryBase;
    virtual ~ChainedOptimizerFactory() = default;
    expected<std::unique_ptr<DiscreteGmOptimizerBase>> create(const DiscreteGm &gm,
                                                              OptimizerParameters &&params) const override
    {
        return std::make_unique<ChainedOptimizers>(gm, std::move(params));
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
        return "Chain multiple optizer st. optimzers are warm started with the best solution of the previous "
               "optimizer(s).";
    }
};

} // namespace nxtgm

XPLUGIN_CREATE_XPLUGIN_FACTORY(nxtgm::ChainedOptimizerFactory);

namespace nxtgm
{
ChainedOptimizers::ChainedOptimizers(const DiscreteGm &gm, OptimizerParameters &&parameters)
    : base_type(gm, parameters),
      parameters_(parameters),
      best_solution_value_(),
      best_solution_(gm.num_variables(), 0),
      is_partial_optimal_(gm.num_variables(), 0),
      _current_optimizer(nullptr)
{

    ensure_all_handled(name(), parameters);
    best_solution_value_ = gm.evaluate(best_solution_, false /* early exit when infeasible*/);
}

class ReporterCallbackForwarder : public DiscreteGmOptimizerBase::reporter_callback_base_type
{
  public:
    using base_type = typename DiscreteGmOptimizerBase::reporter_callback_base_type;

    ReporterCallbackForwarder(DiscreteGmOptimizerBase *optimizer, base_type *callback)
        : base_type(optimizer),
          callback(callback)
    {
    }

    virtual void begin()
    {
        // this->callback->begin();
    }
    virtual bool report()
    {
        return this->callback->report();
    }
    virtual bool report_data(const ReportData &data)
    {
        return this->callback->report_data(data);
    }
    virtual void end()
    {
        // this->callback->end();
    }

  private:
    base_type *callback;
};

OptimizationStatus ChainedOptimizers::optimize_impl(reporter_callback_wrapper_type &reporter_callback,
                                                    repair_callback_wrapper_type & /*repair_callback not used*/,
                                                    const_discrete_solution_span starting_point)
{
    // shortcut to the model
    const auto &gm = this->model();

    OptimizationStatus total_status = OptimizationStatus::CONVERGED;

    const std::size_t num_optimizers = parameters_.optimizer_names.size();

    if (starting_point.size() > 0)
    {
        std::copy(starting_point.begin(), starting_point.end(), best_solution_.begin());
        best_solution_value_ = gm.evaluate(best_solution_, false /* early exit when infeasible*/);
    }

    std::cout << "num_optimizers: " << num_optimizers << std::endl;

    for (std::size_t opt_index = 0; opt_index < num_optimizers; ++opt_index)
    {
        const auto &optimizer_name = parameters_.optimizer_names[opt_index];
        const auto &optimizer_parameters = parameters_.optimizer_parameters[opt_index];

        std::cout << "optimizer_name: " << optimizer_name << std::endl;

        auto expected_optimizer = discrete_gm_optimizer_factory(gm, optimizer_name, optimizer_parameters);
        if (!expected_optimizer)
        {
            throw std::runtime_error(expected_optimizer.error());
        }
        auto optimizer = std::move(expected_optimizer.value());
        _current_optimizer = optimizer.get();

        auto starting_point = const_discrete_solution_span(best_solution_.data(), best_solution_.size());

        OptimizationStatus status;
        if (reporter_callback)
        {
            ReporterCallbackForwarder reporter_callback_forwarder(optimizer.get(), reporter_callback.get());
            status = optimizer->optimize(&reporter_callback_forwarder, nullptr, starting_point);
        }
        else
        {
            status = optimizer->optimize(nullptr, nullptr, starting_point);
        }
        _current_optimizer = nullptr;

        auto solution = optimizer->best_solution();
        auto solution_value = gm.evaluate(solution, false /* early exit when infeasible*/);

        if (auto solution_value = optimizer->best_solution_value(); solution_value < best_solution_value_)
        {
            best_solution_value_ = solution_value;
            best_solution_ = optimizer->best_solution();

            if (!this->report(reporter_callback))
            {
                std::cout << "callback exit" << std::endl;
                return OptimizationStatus::CALLBACK_EXIT;
            }
        }
        if (this->time_limit_reached())
        {
            return OptimizationStatus::TIME_LIMIT_REACHED;
        }
        if (status == OptimizationStatus::OPTIMAL || status == OptimizationStatus::INFEASIBLE)
        {
            return status;
        }
        else if (status == OptimizationStatus::LOCAL_OPTIMAL)
        {
            total_status = status;
        }
    }

    return total_status;
}

SolutionValue ChainedOptimizers::best_solution_value() const
{
    return _current_optimizer == nullptr ? best_solution_value_ : _current_optimizer->best_solution_value();
}
SolutionValue ChainedOptimizers::current_solution_value() const
{
    return _current_optimizer == nullptr ? best_solution_value_ : _current_optimizer->current_solution_value();
}

const typename ChainedOptimizers::solution_type &ChainedOptimizers::best_solution() const
{
    return _current_optimizer == nullptr ? best_solution_ : _current_optimizer->best_solution();
}
const typename ChainedOptimizers::solution_type &ChainedOptimizers::current_solution() const
{
    return _current_optimizer == nullptr ? best_solution_ : _current_optimizer->current_solution();
}

bool ChainedOptimizers::is_partial_optimal(std::size_t variable_index) const
{
    return is_partial_optimal_[variable_index];
}

} // namespace nxtgm
