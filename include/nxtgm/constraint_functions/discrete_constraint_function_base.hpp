#pragma once

#include <nlohmann/json.hpp>
#include <nxtgm/nxtgm.hpp>
#include <nxtgm/utils/lp.hpp>
#include <nxtgm/utils/serialize.hpp>

namespace nxtgm
{

struct IlpConstraintBuilderBuffer
{
    void ensure_size(std::size_t max_constraint_size, std::size_t max_constraint_arity);
    std::vector<energy_type> how_violated_buffer;
    std::vector<discrete_label_type> label_buffer;
    std::vector<discrete_label_type> shape_buffer;
};

class DiscreteConstraintFunctionBase
{
  public:
    virtual ~DiscreteConstraintFunctionBase() = default;

    virtual std::size_t arity() const = 0;
    virtual discrete_label_type shape(std::size_t index) const = 0;

    virtual energy_type how_violated(const discrete_label_type *discrete_labels) const = 0;

    // convenience function
    virtual std::size_t size() const;
    virtual energy_type how_violated(std::initializer_list<discrete_label_type> labels) const;
    virtual std::unique_ptr<DiscreteConstraintFunctionBase> clone() const = 0;

    virtual void add_to_lp(IlpData &ilp_data, const std::size_t *indicator_variables_mapping) const;
    virtual nlohmann::json serialize_json() const = 0;

    virtual std::unique_ptr<DiscreteConstraintFunctionBase> bind(
        const span<std::size_t> &binded_vars, const span<discrete_label_type> &binded_vars_labels) const;

    virtual void serialize(Serializer &serializer) const = 0;
};

// helper class to have a shape object
// with operator[] and size()
class DiscreteConstraintFunctionShape
{
  public:
    inline DiscreteConstraintFunctionShape(const DiscreteConstraintFunctionBase *function)
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
    const DiscreteConstraintFunctionBase *function_;
};

using DiscretConstraintFunctionSerializationFactory =
    std::unordered_map<std::string,
                       std::function<std::unique_ptr<DiscreteConstraintFunctionBase>(const nlohmann::json &)>>;

std::unique_ptr<DiscreteConstraintFunctionBase> discrete_constraint_function_deserialize_json(
    const nlohmann::json &json, const DiscretConstraintFunctionSerializationFactory &user_factory =
                                    DiscretConstraintFunctionSerializationFactory());

std::unique_ptr<DiscreteConstraintFunctionBase> discrete_constraint_function_deserialize(Deserializer &deserializer);
} // namespace nxtgm
