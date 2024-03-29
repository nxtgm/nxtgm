#pragma once

#include <nxtgm/models/gm/discrete_gm.hpp>
#include <nxtgm/optimizers/callback_base.hpp>
#include <nxtgm/optimizers/optimizer_base.hpp>

#include <nlohmann/json.hpp>
#include <nxtgm/plugins/plugin_registry.hpp>
#include <xplugin/xfactory.hpp>

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

    virtual bool is_partial_optimal(std::size_t variable_index) const;
};

class DiscreteGmOptimizerFactoryBase
{
  public:
    virtual ~DiscreteGmOptimizerFactoryBase() = default;

    virtual expected<std::unique_ptr<DiscreteGmOptimizerBase>> create(const DiscreteGm &gm,
                                                                      OptimizerParameters &&params) const = 0;

    virtual std::unique_ptr<DiscreteGmOptimizerBase> create_unique(const DiscreteGm &gm,
                                                                   OptimizerParameters &&params) const;

    static std::string plugin_type();

    static std::string plugin_dir_env_var();

    // priority of the plugin (higher means more important)
    virtual int priority() const = 0;

    // license of the plugin
    virtual std::string license() const = 0;

    // description of the plugin
    virtual std::string description() const = 0;

    // get the registry of the plugin
    static plugin_registry<DiscreteGmOptimizerFactoryBase> &get_registry();
};

} // namespace nxtgm
