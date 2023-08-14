#pragma once

#include <nxtgm/models/gm/discrete_gm.hpp>
#include <nxtgm/optimizers/optimizer_base.hpp>

#include <nlohmann/json.hpp>

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

class DiscreteOptimizerFactoryBase
{
  public:
    virtual ~DiscreteOptimizerFactoryBase() = default;

    virtual std::unique_ptr<DiscreteGmOptimizerBase> create(const DiscreteGm &gm) const = 0;
    virtual std::unique_ptr<DiscreteOptimizerFactoryBase> clone() const = 0;
};

} // namespace nxtgm
