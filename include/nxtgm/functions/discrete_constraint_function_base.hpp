#pragma once

#include <nlohmann/json.hpp>
#include <nxtgm/functions/discrete_function_base.hpp>
#include <nxtgm/nxtgm.hpp>
#include <nxtgm/utils/lp.hpp>
#include <nxtgm/utils/serialize.hpp>

namespace nxtgm
{

class DiscreteConstraintFunctionBase : public DiscreteFunctionBase
{
  public:
    virtual ~DiscreteConstraintFunctionBase() = default;

    virtual std::unique_ptr<DiscreteConstraintFunctionBase> clone() const = 0;

    virtual void add_to_lp(IlpData &ilp_data, const std::size_t *indicator_variables_mapping) const;

    virtual std::unique_ptr<DiscreteConstraintFunctionBase> bind(
        const span<std::size_t> &binded_vars, const span<discrete_label_type> &binded_vars_labels) const;
};

using DiscretConstraintFunctionSerializationFactory =
    std::unordered_map<std::string,
                       std::function<std::unique_ptr<DiscreteConstraintFunctionBase>(const nlohmann::json &)>>;

std::unique_ptr<DiscreteConstraintFunctionBase> discrete_constraint_function_deserialize_json(
    const nlohmann::json &json, const DiscretConstraintFunctionSerializationFactory &user_factory =
                                    DiscretConstraintFunctionSerializationFactory());

std::unique_ptr<DiscreteConstraintFunctionBase> discrete_constraint_function_deserialize(Deserializer &deserializer);
} // namespace nxtgm
