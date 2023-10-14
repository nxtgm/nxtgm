#pragma once

#include <initializer_list>

#include <nlohmann/json.hpp>
#include <nxtgm/functions/discrete_function_base.hpp>
#include <nxtgm/nxtgm.hpp>
#include <nxtgm/utils/lp.hpp>
#include <nxtgm/utils/serialize.hpp>

namespace nxtgm
{

class DiscreteEnergyFunctionBase;

class DiscreteEnergyFunctionBase : public DiscreteFunctionBase
{
  public:
    virtual ~DiscreteEnergyFunctionBase() = default;

    virtual void add_to_lp(IlpData &ilp_data, const std::size_t *indicator_variables_mapping) const;

    virtual void compute_factor_to_variable_messages(const energy_type *const *in_messages,
                                                     energy_type **out_messages) const;

    virtual std::unique_ptr<DiscreteEnergyFunctionBase> clone() const = 0;

    virtual std::unique_ptr<DiscreteEnergyFunctionBase> bind(span<const std::size_t> binded_vars,
                                                             span<const discrete_label_type> binded_vars_labels) const;
};

using DiscretEnergyFunctionSerializationFactory =
    std::unordered_map<std::string, std::function<std::unique_ptr<DiscreteEnergyFunctionBase>(const nlohmann::json &)>>;

std::unique_ptr<DiscreteEnergyFunctionBase> discrete_energy_function_deserialize_json(
    const nlohmann::json &json,
    const DiscretEnergyFunctionSerializationFactory &user_factory = DiscretEnergyFunctionSerializationFactory());

std::unique_ptr<DiscreteEnergyFunctionBase> discrete_energy_function_deserialize(Deserializer &deserializer);

} // namespace nxtgm
