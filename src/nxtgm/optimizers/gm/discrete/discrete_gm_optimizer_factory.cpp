#include <nxtgm/optimizers/gm/discrete/discrete_gm_optimizer_factory.hpp>
#include <nxtgm/plugins/plugin_registry.hpp>

#include <iostream>

namespace nxtgm
{
std::unique_ptr<DiscreteGmOptimizerBase> discrete_gm_optimizer_factory(const DiscreteGm &gm, const std::string &name,
                                                                       const nlohmann::json parameter)
{
    return get_plugin_registry<DiscreteGmOptimizerFactoryBase>().get_factory(name)->create(gm, parameter);
}

} // namespace nxtgm
