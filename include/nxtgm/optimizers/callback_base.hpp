#pragma once

// tuple
#include <tuple>

#include <nxtgm/models/solution_value.hpp>
#include <nxtgm/utils/uany.hpp>

namespace nxtgm
{
template <class OPTIMIZER_BASE_TYPE>
class CallbackBase
{
  public:
    using optimizer_base_type = OPTIMIZER_BASE_TYPE;

    CallbackBase(const optimizer_base_type *optimizer)
        : optimizer_(optimizer)
    {
    }

    virtual ~CallbackBase() = default;

    // virtual operator()
    virtual const optimizer_base_type *optimizer() const
    {
        return this->optimizer_;
    }

  private:
    const optimizer_base_type *optimizer_;
};

struct ReportData
{
    std::unordered_map<std::string, span<const double>> double_data;
    std::unordered_map<std::string, span<const int64_t>> int_data;
    std::unordered_map<std::string, uany> any_data;
};

// print / log energy / bounds / etc
template <class OPTIMIZER_BASE_TYPE>
class ReporterCallbackBase : public CallbackBase<OPTIMIZER_BASE_TYPE>
{
  public:
    using optimizer_base_type = OPTIMIZER_BASE_TYPE;

    ReporterCallbackBase(const optimizer_base_type *optimizer)
        : CallbackBase<OPTIMIZER_BASE_TYPE>(optimizer)
    {
    }

    virtual ~ReporterCallbackBase() = default;

    virtual void begin()
    {
    }
    virtual bool report() = 0;
    virtual bool report_data(const ReportData &data)
    {
        return this->report();
    }
    virtual void end()
    {
    }
};

template <class REPORTER_CALLBACK_BASE_TYPE>
class ReporterCallbackWrapper
{
  public:
    using reporter_callback_base_type = REPORTER_CALLBACK_BASE_TYPE;
    ReporterCallbackWrapper(reporter_callback_base_type *reporter_callback)
        : reporter_callback_(reporter_callback)
    {
        if (this->reporter_callback_ != nullptr)
        {
            this->reporter_callback_->begin();
        }
    }
    ~ReporterCallbackWrapper()
    {
        if (this->reporter_callback_ != nullptr)
        {
            this->reporter_callback_->end();
        }
    }
    inline bool report()
    {
        if (this->reporter_callback_ != nullptr)
        {
            return this->reporter_callback_->report();
        }
        return true;
    }

    inline bool report(const ReportData &data)
    {
        if (this->reporter_callback_ != nullptr)
        {
            return this->reporter_callback_->report_data(data);
        }
        return true;
    }

    // convert to bool
    inline operator bool() const
    {
        return this->reporter_callback_ != nullptr;
    }
    reporter_callback_base_type *get() const
    {
        return this->reporter_callback_;
    }

  private:
    reporter_callback_base_type *reporter_callback_;
};

// repair solutions with violated constraints
// improve solutions with better energy
template <class OPTIMIZER_BASE_TYPE>
class RepairCallbackBase : public CallbackBase<OPTIMIZER_BASE_TYPE>
{
  public:
    using optimizer_base_type = OPTIMIZER_BASE_TYPE;
    using model_type = typename optimizer_base_type::model_type;
    using solution_type = typename optimizer_base_type::solution_type;
    RepairCallbackBase(const optimizer_base_type *optimizer)
        : CallbackBase<OPTIMIZER_BASE_TYPE>(optimizer)
    {
    }

    virtual ~RepairCallbackBase() = default;

    virtual bool repair(solution_type &solution, SolutionValue &solution_eval)
    {
        return false;
    }
    virtual bool improve(solution_type &solution, SolutionValue &solution_eval)
    {
        return false;
    }
};

template <class REPAIR_CALLBACK_BASE_TYPE>
class RepairCallbackWrapper
{
  public:
    using repair_callback_base_type = REPAIR_CALLBACK_BASE_TYPE;
    using optimizer_base_type = typename repair_callback_base_type::optimizer_base_type;

    using solution_type = typename optimizer_base_type::solution_type;

    RepairCallbackWrapper(repair_callback_base_type *repair_callback)
        : repair_callback_(repair_callback)
    {
    }
    inline bool repair(solution_type &solution, SolutionValue &solution_eval)
    {
        if (this->repair_callback_ != nullptr)
        {
            return this->repair_callback_->repair(solution, solution_eval);
        }
        return false;
    }
    inline bool improve(solution_type &solution, SolutionValue &solution_eval)
    {
        if (this->repair_callback_ != nullptr)
        {
            return this->repair_callback_->improve(solution, solution_eval);
        }
        return false;
    }

    // convert to bool
    inline operator bool() const
    {
        return this->repair_callback_ != nullptr;
    }

  private:
    repair_callback_base_type *repair_callback_;
};

} // namespace nxtgm
