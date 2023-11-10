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
    OptimalForFirstOrderMatching = 1 << 6,
    MetaOptimizer = 1 << 7
};

// custom exception UnsupportedModelException
class UnsupportedModelException : public std::exception
{
  public:
    UnsupportedModelException(const std::string &message)
        : message_(message)
    {
    }

    virtual const char *what() const throw()
    {
        return message_.c_str();
    }

  private:
    std::string message_;
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

    inline OptimizerBase(const model_type &model, OptimizerParameters &parameters)
        : model_(model),
          timer_(),
          time_limit_(std::chrono::duration<double>::max())
    {
        if (auto it = parameters.int_parameters.find("time_limit_ms"); it != parameters.int_parameters.end())
        {
            time_limit_ = std::chrono::milliseconds(it->second);
            parameters.int_parameters.erase(it);
        }
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
        timer_.start();
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

  public:
    std::chrono::duration<double> remaining_time() const
    {
        return time_limit_ - timer_.elapsed();
    }

    bool time_limit_reached() const
    {
        return timer_.elapsed() > time_limit_;
    }
    std::chrono::duration<double> get_time_limit() const
    {
        return time_limit_;
    }
    void set_time_limit(std::chrono::duration<double> time_limit)
    {
        time_limit_ = time_limit;
    }

    bool report(reporter_callback_wrapper_type &reporter_callback)
    {
        timer_.pause();
        const auto ret = reporter_callback.report();
        timer_.resume();
        return ret;
    }

  private:
    const model_type &model_;
    Timer timer_;
    std::chrono::duration<double> time_limit_;
};
} // namespace nxtgm
