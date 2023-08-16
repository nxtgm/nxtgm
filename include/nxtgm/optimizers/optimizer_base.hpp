#pragma once

#include <chrono>
#include <limits>
#include <tuple>

#include <nxtgm/optimizers/callback_base.hpp>

#define NXTGM_OPTIMIZER_DEFAULT_FACTORY(NAME)                                                                          \
    class NAME##DiscreteGmOptimizerFactory : public DiscreteGmOptimizerFactoryBase                                     \
    {                                                                                                                  \
      public:                                                                                                          \
        using factory_base_type = DiscreteGmOptimizerFactoryBase;                                                      \
        virtual ~NAME##DiscreteGmOptimizerFactory() = default;                                                         \
        std::unique_ptr<DiscreteGmOptimizerBase> create(const DiscreteGm &gm,                                          \
                                                        const nlohmann::json &params) const override                   \
        {                                                                                                              \
            return std::make_unique<NAME>(gm, params);                                                                 \
        }                                                                                                              \
        int priority() const override                                                                                  \
        {                                                                                                              \
            return 1;                                                                                                  \
        }                                                                                                              \
        std::string license() const override                                                                           \
        {                                                                                                              \
            return "Mit";                                                                                              \
        }                                                                                                              \
        std::string description() const override                                                                       \
        {                                                                                                              \
            return #NAME;                                                                                              \
        }                                                                                                              \
    };

namespace nxtgm
{
enum class OptimizationStatus
{
    OPTIMAL,
    PARTIAL_OPTIMAL,
    LOCAL_OPTIMAL,
    INFEASIBLE,
    UNKNOWN,
    TIME_LIMIT_REACHED,
    ITERATION_LIMIT_REACHED,
    CONVERGED,
    CALLBACK_EXIT
};

class OptimizerParametersBase
{
  public:
    inline OptimizerParametersBase(const nlohmann::json &json_parameters)
    {
        if (json_parameters.contains("time_limit"))
        {
            time_limit = std::chrono::seconds(json_parameters["time_limit"].get<int>());
        }
    }

    std::chrono::duration<double> time_limit = std::chrono::duration<double>::max();
};

template <class MODEL_TYPE, class DERIVED_OPTIMIZER_BASE>
class OptimizerBase
{

  public:
    using self_type = OptimizerBase<MODEL_TYPE, DERIVED_OPTIMIZER_BASE>;
    using model_type = MODEL_TYPE;

    using solution_type = typename model_type::solution_type;

    using reporter_callback_base_type = ReporterCallbackBase<DERIVED_OPTIMIZER_BASE>;
    using reporter_callback_wrapper_type = ReporterCallbackWrapper<reporter_callback_base_type>;

    using repair_callback_base_type = RepairCallbackBase<self_type>;
    using repair_callback_wrapper_type = RepairCallbackWrapper<repair_callback_base_type>;

    inline OptimizerBase(const model_type &model)
        : model_(model)
    {
    }

    virtual ~OptimizerBase() = default;
    virtual energy_type lower_bound() const
    {
        return -std::numeric_limits<energy_type>::infinity();
    }

    virtual SolutionValue best_solution_value() const
    {
        return this->model().evaluate(this->best_solution(), false /* early exit when infeasible*/);
    }

    virtual SolutionValue current_solution_value() const
    {
        return this->model().evaluate(this->best_solution(), false /* early exit when infeasible*/);
    }

    virtual OptimizationStatus optimize(reporter_callback_base_type *reporter_callback = nullptr,
                                        repair_callback_base_type *repair_callback = nullptr,
                                        const_discrete_solution_span starting_point = const_discrete_solution_span())
    {
        reporter_callback_wrapper_type reporter_callback_wrapper(reporter_callback);
        repair_callback_wrapper_type repair_callback_wrapper(repair_callback);

        return this->optimize(reporter_callback_wrapper, repair_callback_wrapper, starting_point);
    }

    virtual OptimizationStatus optimize(reporter_callback_wrapper_type &reporter_callback,
                                        repair_callback_wrapper_type &repair_callback,
                                        const_discrete_solution_span starting_point) = 0;

    virtual const model_type &model() const
    {
        return this->model_;
    }
    virtual const solution_type &best_solution() const = 0;
    virtual const solution_type &current_solution() const = 0;

  private:
    const model_type &model_;
};
} // namespace nxtgm
