#pragma once

#include <nxtgm/models/gm/discrete_gm.hpp>
#include <nxtgm/optimizers/callback_base.hpp>
#include <nxtgm/optimizers/optimizer_base.hpp>

#include <nlohmann/json.hpp>
#include <xplugin/xfactory.hpp>

#include <iostream>
#include <memory>
#include <nxtgm/plugins/plugin_priority.hpp>
#include <string>
#include <unordered_map>

namespace nxtgm
{

class DiscreteGmOptimizerBase : public OptimizerBase<DiscreteGm, DiscreteGmOptimizerBase>
{
  public:
    using base_type = OptimizerBase<DiscreteGm, DiscreteGmOptimizerBase>;
    using base_type::base_type;
    using base_type::optimize;

    virtual ~DiscreteGmOptimizerBase() = default;

    virtual bool is_partial_optimal(std::size_t variable_index) const
    {
        return false;
    }
};

class DiscreteGmOptimizerFactoryBase
{
  public:
    virtual ~DiscreteGmOptimizerFactoryBase() = default;

    virtual std::unique_ptr<DiscreteGmOptimizerBase> create(const DiscreteGm &gm,
                                                            const OptimizerParameters &params) const = 0;

    static std::string plugin_type()
    {
        return "discrete_gm_optimizer";
    }

    static std::string plugin_dir_env_var()
    {
        return "NXTGM_DISCRETE_GM_OPTIMIZER_PLUGIN_PATH";
    }

    // priority of the plugin (higher means more important)
    virtual int priority() const = 0;

    // license of the plugin
    virtual std::string license() const = 0;

    // description of the plugin
    virtual std::string description() const = 0;

    // flags of the plugin
    virtual OptimizerFlags flags() const = 0;
};

} // namespace nxtgm
