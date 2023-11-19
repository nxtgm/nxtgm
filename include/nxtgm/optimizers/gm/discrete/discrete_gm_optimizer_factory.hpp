#pragma once

#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>

namespace nxtgm
{

expected<std::unique_ptr<DiscreteGmOptimizerBase>> discrete_gm_optimizer_factory(const DiscreteGm &gm,
                                                                                 const std::string &name,
                                                                                 const OptimizerParameters &parameter);

expected<std::unique_ptr<DiscreteGmOptimizerBase>> discrete_gm_optimizer_factory(
    const DiscreteGm &gm, const std::string &name, OptimizerParameters &&parameter = OptimizerParameters());

DiscreteGmOptimizerFactoryBase *get_discrete_gm_optimizer_factory(const std::string &name);

} // namespace nxtgm
