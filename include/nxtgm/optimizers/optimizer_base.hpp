#pragma once

#include <chrono>
#include <limits>
#include <nxtgm/optimizers/callback_base.hpp>
#include <nxtgm/optimizers/optimizer_parameters.hpp>
#include <nxtgm/utils/timer.hpp>
#include <tuple>

namespace nxtgm
{

enum class OptimizerFlags : uint64_t
{
    None = 0,
    WarmStartable = 1 << 0,
    Optimal = 1 << 1,
    PartialOptimal = 1 << 2,
    LocalOptimal = 1 << 3,
    OptimalOnTrees = 1 << 4,
    OptimalOnBinarySecondOrderSubmodular = 1 << 5,
    MetaOptimizer = 1 << 6
};

inline OptimizerFlags operator|(OptimizerFlags lhs, OptimizerFlags rhs)
{
    return static_cast<OptimizerFlags>(static_cast<std::underlying_type<OptimizerFlags>::type>(lhs) |
                                       static_cast<std::underlying_type<OptimizerFlags>::type>(rhs));
}

inline OptimizerFlags operator&(OptimizerFlags lhs, OptimizerFlags rhs)
{
    return static_cast<OptimizerFlags>(static_cast<std::underlying_type<OptimizerFlags>::type>(lhs) &
                                       static_cast<std::underlying_type<OptimizerFlags>::type>(rhs));
}

class OptimizerTimer : public AutoStartedTimer
{
  public:
    using AutoStartedTimer::AutoStartedTimer;

    template <class REPORTER_CALLBACK>
    bool begin(REPORTER_CALLBACK &&reporter_callback)
    {
        if (!reporter_callback)
        {
            return true;
        }

        return this->paused_call([&]() { reporter_callback->begin(); });
    }
    template <class REPORTER_CALLBACK>
    bool end(REPORTER_CALLBACK &&reporter_callback)
    {
        if (!reporter_callback)
        {
            return true;
        }

        return this->paused_call([&]() { reporter_callback->end(); });
    }
    template <class REPORTER_CALLBACK>
    bool report(REPORTER_CALLBACK &&reporter_callback)
    {
        if (!reporter_callback)
        {
            return true;
        }

        return this->paused_call([&]() { reporter_callback->report(); });
    }
};

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
    inline OptimizerParametersBase(const OptimizerParameters &p)
    {

        if (auto it = p.int_parameters.find("time_limit_ms"); it != p.int_parameters.end())
        {
            time_limit = std::chrono::milliseconds(it->second);
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

    virtual bool is_warm_startable() const
    {
        return false;
    }

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

    OptimizationStatus optimize(reporter_callback_base_type *reporter_callback = nullptr,
                                repair_callback_base_type *repair_callback = nullptr,
                                const_discrete_solution_span starting_point = const_discrete_solution_span())
    {
        // this calls begin / end in the constructor / destructor
        reporter_callback_wrapper_type reporter_callback_wrapper(reporter_callback);
        repair_callback_wrapper_type repair_callback_wrapper(repair_callback);

        return this->optimize_impl(reporter_callback_wrapper, repair_callback_wrapper, starting_point);
    }

    virtual const model_type &model() const
    {
        return this->model_;
    }
    virtual const solution_type &best_solution() const = 0;
    virtual const solution_type &current_solution() const = 0;

  protected:
    virtual OptimizationStatus optimize_impl(reporter_callback_wrapper_type &reporter_callback,
                                             repair_callback_wrapper_type &repair_callback,
                                             const_discrete_solution_span starting_point) = 0;

  private:
    const model_type &model_;
};
} // namespace nxtgm
