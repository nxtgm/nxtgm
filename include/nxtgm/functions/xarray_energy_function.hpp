#pragma once

#include <algorithm>
#include <cstdint>
#include <nxtgm/functions/discrete_energy_function_base.hpp>
#include <nxtgm/nxtgm.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>

namespace nxtgm
{

class XArray : public DiscreteEnergyFunctionBase
{
  public:
    using base_type = DiscreteEnergyFunctionBase;
    using base_type::value;

    using xarray_type = xt::xarray<energy_type>;
    inline static std::string serialization_key()
    {
        return "array";
    }
    XArray() = default;
    template <class TENSOR>
    XArray(TENSOR &&values)
        : values_(std::forward<TENSOR>(values))
    {
    }

    XArray(const xarray_type &values);
    discrete_label_type shape(std::size_t index) const override;

    std::size_t arity() const override;

    std::size_t size() const override;

    energy_type value(const discrete_label_type *discrete_labels) const override;
    std::unique_ptr<DiscreteEnergyFunctionBase> clone() const override;

    void copy_values(energy_type *energies) const override;
    void add_values(energy_type *energies) const override;
    static std::unique_ptr<DiscreteEnergyFunctionBase> deserialize_json(const nlohmann::json &json);
    static std::unique_ptr<DiscreteEnergyFunctionBase> deserialize(Deserializer &deserializer);
    nlohmann::json serialize_json() const override;
    void serialize(Serializer &serializer) const override;

  private:
    xarray_type values_;
};

} // namespace nxtgm
