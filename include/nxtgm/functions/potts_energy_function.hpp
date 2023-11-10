#pragma once

#include <nxtgm/functions/discrete_energy_function_base.hpp>
#include <nxtgm/nxtgm.hpp>

#include <algorithm>
#include <cstdint>

namespace nxtgm
{

class Potts : public DiscreteEnergyFunctionBase
{
  public:
    using base_type = DiscreteEnergyFunctionBase;
    using base_type::value;

    inline static std::string serialization_key()
    {
        return "potts";
    }
    Potts() = default;
    Potts(std::size_t num_labels, energy_type beta);

    std::size_t arity() const override;
    discrete_label_type shape(std::size_t) const override;
    std::size_t size() const override;
    energy_type value(const discrete_label_type *discrete_labels) const override;
    std::unique_ptr<DiscreteEnergyFunctionBase> clone() const override;

    void copy_values(energy_type *energies) const override;
    void add_values(energy_type *energies) const override;

    static std::unique_ptr<DiscreteEnergyFunctionBase> deserialize_json(const nlohmann::json &json);
    static std::unique_ptr<DiscreteEnergyFunctionBase> deserialize(Deserializer &deserializer);
    nlohmann::json serialize_json() const override;
    void serialize(Serializer &serializer) const override;

    void compute_to_variable_messages(const energy_type *const *in_messages, energy_type **out_messages) const override;

    std::unique_ptr<DiscreteEnergyFunctionBase> bind(span<const std::size_t> binded_vars,
                                                     span<const discrete_label_type> binded_vars_labels) const override;

  private:
    std::size_t num_labels_;
    energy_type beta_;
};

} // namespace nxtgm
