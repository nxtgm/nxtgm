#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>

namespace nxtgm
{
bool DiscreteGmOptimizerBase::is_partial_optimal(std::size_t variable_index) const
{
    return false;
}

std::string DiscreteGmOptimizerFactoryBase::plugin_type()
{
    return "discrete_gm_optimizer";
}

std::string DiscreteGmOptimizerFactoryBase::plugin_dir_env_var()
{
    return "NXTGM_DISCRETE_GM_OPTIMIZER_PLUGIN_PATH";
}

plugin_registry<DiscreteGmOptimizerFactoryBase> &DiscreteGmOptimizerFactoryBase::get_registry()
{
    static plugin_registry<DiscreteGmOptimizerFactoryBase> registry;
    return registry;
}

std::unique_ptr<DiscreteGmOptimizerBase> DiscreteGmOptimizerFactoryBase::create_unique(
    const DiscreteGm &gm, OptimizerParameters &&params) const
{
    return create(gm, std::move(params)).value();
}

} // namespace nxtgm
