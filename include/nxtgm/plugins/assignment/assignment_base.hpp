#pragma once
#include <cstddef>
#include <memory>
#include <string>

#include <nxtgm/nxtgm.hpp>
#include <nxtgm/optimizers/optimizer_parameters.hpp>
#include <nxtgm/plugins/plugin_factory_base.hpp>
#include <nxtgm/plugins/plugin_priority.hpp>
#include <nxtgm/plugins/plugin_registry.hpp>

namespace nxtgm
{

class AssignmentBase
{
  public:
    AssignmentBase() = default;
    virtual ~AssignmentBase() = default;

    virtual OptimizationStatus optimize(const int *starting_point) = 0;
    virtual OptimiztionStatus optimize_min_marginals(xt::xtensor<double, 2> &min_marginals) = 0;
};

class AssignmentFactoryBase : public PluginFactoryBase
{

  public:
    virtual ~AssignmentFactoryBase() = default;

    inline static std::string plugin_type()
    {
        return "assignment";
    }

    inline static std::string plugin_dir_env_var()
    {
        return "NXTGM_ASSIGNMENT_PLUGIN_PATH";
    }

    // create an instance of the plugin
    virtual std::unique_ptr<AssignmentBase> create(xt::xtensor<double, 2> &&costs, bool with_ignore_label,
                                                   OptimizerParameters &&parameters) = 0;

    static plugin_registry<AssignmentFactoryBase> &get_registry();
};

} // namespace nxtgm
