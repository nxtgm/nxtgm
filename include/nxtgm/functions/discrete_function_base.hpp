#pragma once

#include <initializer_list>

#include <nlohmann/json.hpp>
#include <nxtgm/nxtgm.hpp>
#include <nxtgm/utils/lp.hpp>
#include <nxtgm/utils/serialize.hpp>

namespace nxtgm
{

class DiscreteFunctionBase;

class DiscreteFunctionBase
{
  public:
    virtual ~DiscreteFunctionBase() = default;

    // pure virtual functions:
    virtual std::size_t arity() const = 0;
    virtual discrete_label_type shape(std::size_t index) const = 0;
    virtual energy_type value(const discrete_label_type *discrete_labels) const = 0;

    // // function with default implementation:
    virtual std::size_t size() const;
    virtual energy_type value(std::initializer_list<discrete_label_type> discrete_labels) const;

    virtual void copy_values(energy_type *energies) const;
    virtual void copy_shape(discrete_label_type *shape) const;
    virtual void add_values(energy_type *energies) const;

    // virtual void add_to_lp(IlpData &ilp_data, const std::size_t *indicator_variables_mapping) const;

    // virtual void compute_factor_to_variable_messages(const energy_type *const *in_messages,
    //                                                  energy_type **out_messages) const;

    // virtual std::unique_ptr<DiscreteFunctionBase> clone() const = 0;

    // virtual std::unique_ptr<DiscreteFunctionBase> bind(span<const std::size_t> binded_vars,
    //                                                          span<const discrete_label_type> binded_vars_labels)
    //                                                          const;

    virtual nlohmann::json serialize_json() const = 0;
    virtual void serialize(Serializer &serializer) const = 0;
};

// helper class to have a shape object
// with operator[] and size()
class DiscreteFunctionShape
{
  public:
    inline DiscreteFunctionShape(const DiscreteFunctionBase *function)
        : function_(function)
    {
    }

    inline std::size_t size() const
    {
        return function_->arity();
    }
    inline discrete_label_type operator[](std::size_t index) const
    {
        return function_->shape(index);
    }

  private:
    const DiscreteFunctionBase *function_;
};

} // namespace nxtgm
