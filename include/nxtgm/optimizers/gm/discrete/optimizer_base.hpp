#pragma once

#include <nxtgm/models/gm/discrete_gm.hpp>
#include <nxtgm/optimizers/optimizer_base.hpp>

namespace nxtgm
{

class DiscreteGmOptimizerBase : public OptimizerBase<DiscreteGm, DiscreteGmOptimizerBase>
{
  public:
    using base_type = OptimizerBase<DiscreteGm, DiscreteGmOptimizerBase>;
    using base_type::base_type;
    using base_type::optimize;

    virtual bool is_partial_optimal(std::size_t variable_index) const
    {
        return false;
    }
};
} // namespace nxtgm
