#include "nxtgm_test_common.hpp"

#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>
#include <nxtgm/plugins/ilp/ilp_base.hpp>

namespace nxtgm
{
std::vector<std::string> all_optimizers()
{
    std::vector<std::string> result;
    auto &registry = get_plugin_registry<DiscreteGmOptimizerFactoryBase>();
    for (auto &[plugin_name, factory] : registry)
    {
        auto optimizer_name = plugin_name.substr(std::string("discrete_gm_optimizer_").size());
#ifdef _WIN32
        if (optimizer_name == "ilp_based" || optimizer_name == "ilp_highs")
        {
            continue;
        }
#endif
        result.push_back(optimizer_name);
    }
    return result;
}

std::vector<std::string> all_ilp_plugins()
{
    std::vector<std::string> result;
    auto &registry = get_plugin_registry<IlpFactoryBase>();
    for (auto &[plugin_name, factory] : registry)
    {
        auto name = plugin_name.substr(std::string("ilp_").size());
#ifdef _WIN32
        if (name == "highs" || name == "coin_clp")
        {
            continue;
        }
#endif
        result.push_back(name);
    }
    return result;
}

} // namespace nxtgm
