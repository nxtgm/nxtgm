#pragma once

#include <nxtgm/functions/discrete_energy_function_base.hpp>
#include <nxtgm/nxtgm.hpp>
#include <xtensor/xtensor.hpp>
// xtensor adapt
#include <algorithm>
#include <cstdint>
#include <xtensor/xadapt.hpp>

namespace nxtgm
{

template <std::size_t ARITY>
class XTensor : public DiscreteEnergyFunctionBase
{
  public:
    using base_type = DiscreteEnergyFunctionBase;
    using base_type::value;

    using xtensor_type = xt::xtensor<energy_type, ARITY>;
    inline static std::string serialization_key()
    {
        return "array";
    }
    XTensor() = default;
    XTensor(const xtensor_type &values)
        : values_(values)
    {
    }
    template <class TENSOR>
    XTensor(TENSOR &&values)
        : values_(std::forward<TENSOR>(values))
    {
    }

    discrete_label_type shape(std::size_t index) const override
    {
        return values_.shape()[index];
    }

    std::size_t arity() const override
    {
        return ARITY;
    }

    std::size_t size() const override
    {
        return values_.size();
    }

    energy_type value(const discrete_label_type *discrete_labels) const override
    {
        const_discrete_label_span l(discrete_labels, ARITY);
        return values_[l];
    }
    std::unique_ptr<DiscreteEnergyFunctionBase> clone() const override
    {
        return std::make_unique<XTensor<ARITY>>(values_);
    }

    void copy_values(energy_type *energies) const override
    {
        std::copy(values_.data(), values_.data() + values_.size(), energies);
    }
    void add_values(energy_type *energies) const override
    {
        std::transform(values_.data(), values_.data() + values_.size(), energies, energies, std::plus<energy_type>());
    }
    static std::unique_ptr<DiscreteEnergyFunctionBase> deserialize_json(const nlohmann::json &json)
    {

        auto shape = json["shape"].get<std::vector<std::size_t>>();
        auto values = json["values"].get<std::vector<energy_type>>();
        auto values_span = xt::adapt(values, shape);
        return std::make_unique<XTensor<ARITY>>(values_span);
    }
    static std::unique_ptr<DiscreteEnergyFunctionBase> deserialize(Deserializer &deserializer)
    {
        auto f = new XTensor<ARITY>();
        deserializer(f->values_);
        return std::unique_ptr<DiscreteEnergyFunctionBase>(f);
    }

    nlohmann::json serialize_json() const override
    {

        nlohmann::json shape = nlohmann::json::array();
        for (auto s : values_.shape())
        {
            shape.push_back(s);
        }

        // iterator pair to nlhohmann::json
        auto values = nlohmann::json::array();
        for (auto it = values_.begin(); it != values_.end(); ++it)
        {
            values.push_back(*it);
        }

        return {{"type", "array"}, {"dimensions", values_.dimension()}, {"shape", shape}, {"values", values}};
    }
    void serialize(Serializer &serializer) const override
    {
        serializer(values_);
    }

  private:
    xtensor_type values_;
};

using Unary = XTensor<1>;

} // namespace nxtgm
