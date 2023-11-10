#pragma once

#include <nlohmann/json.hpp>
#include <nxtgm/functions/discrete_function_base.hpp>
#include <nxtgm/nxtgm.hpp>
#include <nxtgm/utils/lp.hpp>
#include <nxtgm/utils/serialize.hpp>

namespace nxtgm
{

class Fusion;

class DiscreteConstraintFunctionBase : public DiscreteFunctionBase
{
  public:
    virtual ~DiscreteConstraintFunctionBase() = default;

    virtual std::unique_ptr<DiscreteConstraintFunctionBase> clone() const = 0;

    virtual void add_to_lp(IlpData &ilp_data, const std::size_t *indicator_variables_mapping) const;

    virtual std::unique_ptr<DiscreteConstraintFunctionBase> bind(
        const span<std::size_t> &binded_vars, const span<discrete_label_type> &binded_vars_labels) const;

    virtual void compute_to_variable_messages(const energy_type *const *in_messages, energy_type **out_messages,
                                              energy_type constraint_scaling_factor) const;

    virtual void fuse(const discrete_label_type *labels_a, const discrete_label_type *labels_b,
                      discrete_label_type *labels_ab, const std::size_t fused_arity,
                      const std::size_t *fuse_factor_var_pos, Fusion &fusion) const;
};

using DiscretConstraintFunctionSerializationFactory =
    std::unordered_map<std::string,
                       std::function<std::unique_ptr<DiscreteConstraintFunctionBase>(const nlohmann::json &)>>;

std::unique_ptr<DiscreteConstraintFunctionBase> discrete_constraint_function_deserialize_json(
    const nlohmann::json &json, const DiscretConstraintFunctionSerializationFactory &user_factory =
                                    DiscretConstraintFunctionSerializationFactory());

std::unique_ptr<DiscreteConstraintFunctionBase> discrete_constraint_function_deserialize(Deserializer &deserializer);
} // namespace nxtgm
