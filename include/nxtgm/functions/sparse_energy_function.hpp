#pragma once

#include <nxtgm/functions/discrete_energy_function_base.hpp>
#include <nxtgm/nxtgm.hpp>
#include <nxtgm/utils/sparse_array.hpp>

namespace nxtgm
{

class SparseDiscreteEnergyFunction : public DiscreteEnergyFunctionBase
{
  public:
    using base_type = DiscreteEnergyFunctionBase;
    using base_type::value;

    inline static std::string serialization_key()
    {
        return "sparse";
    }

    template <class SHAPE>
    SparseDiscreteEnergyFunction(SHAPE &&shape)
        : data_(std::forward<SHAPE>(shape))
    {
    }

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
    void add_to_lp(IlpData &ilp_data, const std::size_t *indicator_variables_mapping) const override;

    // not part of the general api
    void set_energy(std::initializer_list<discrete_label_type> labels, energy_type energy);
    void set_energy(const discrete_label_type *labels, energy_type energy);

    void flat_index_to_labels(std::size_t flat_index, discrete_label_type *labels) const;

    auto &data()
    {
        return data_;
    }
    const auto &data() const
    {
        return data_;
    }

  private:
    SparseArray<energy_type> data_;
};

} // namespace nxtgm
