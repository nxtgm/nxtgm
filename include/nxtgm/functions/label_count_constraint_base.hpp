#pragma once

#include <nxtgm/functions/discrete_constraint_function_base.hpp>
#include <nxtgm/nxtgm.hpp>
#include <vector>
#include <xtensor/xarray.hpp>

namespace nxtgm
{

// require a minimum and maximum number how often a label can be used
class LabelCountConstraintBase : public DiscreteConstraintFunctionBase
{
  public:
    // using DiscreteConstraintFunctionBase::value;
    virtual ~LabelCountConstraintBase()
    {
    }

    void add_to_lp(IlpData &ilp_data, const std::size_t *indicator_variables_mapping) const override;
    discrete_label_type shape(std::size_t) const override;
    std::size_t size() const override;

    virtual discrete_label_type num_labels() const = 0;
    virtual std::size_t min_counts(discrete_label_type label) const = 0;
    virtual std::size_t max_counts(discrete_label_type label) const = 0;
};

} // namespace nxtgm
