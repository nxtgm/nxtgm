#pragma once

#include <emscripten.h>
#include <emscripten/val.h>

#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>

namespace nxtgm
{
namespace em = emscripten;
class ReporterCallback : public ReporterCallbackBase<DiscreteGmOptimizerBase>
{
  public:
    using base_type = ReporterCallbackBase<DiscreteGmOptimizerBase>;

    ReporterCallback(const DiscreteGmOptimizerBase *optimizer, em::val callback)
        : base_type(optimizer),
          callback_(callback)
    {
    }

    virtual ~ReporterCallback() = default;

    void begin() override
    {
        callback_["begin"]();
    }
    bool report() override
    {
        return callback_["report"]().as<bool>();
    }
    void end() override
    {
        callback_["end"]();
    }

  private:
    em::val callback_;
};

void export_callbacks()
{
    using reporter_callback_base = ReporterCallbackBase<DiscreteGmOptimizerBase>;
    em::class_<reporter_callback_base>("ReporterCallbackBase")
        .function("begin", &reporter_callback_base::begin)
        .function("report", &reporter_callback_base::report)
        .function("end", &reporter_callback_base::end);

    em::class_<ReporterCallback, em::base<reporter_callback_base>>("ReporterCallback")
        .constructor<DiscreteGmOptimizerBase *, em::val>();
};

} // namespace nxtgm
