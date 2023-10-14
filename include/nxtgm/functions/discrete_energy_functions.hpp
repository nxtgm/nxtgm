#pragma once

#include <nxtgm/functions/discrete_energy_function_base.hpp>
#include <nxtgm/nxtgm.hpp>
#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>
// xtensor adapt
#include <algorithm>
#include <cstdint>
#include <xtensor/xadapt.hpp>

#include <nxtgm/utils/sparse_array.hpp>

#ifndef NXTGM_NO_THREADS
#include <mutex> // std::mutex
#endif

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

    std::unique_ptr<DiscreteEnergyFunctionBase> bind(span<const std::size_t> binded_vars,
                                                     span<const discrete_label_type> binded_vars_labels) const override;

  private:
    std::size_t num_labels_;
    energy_type beta_;
};

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

// label costs
// pay a cost once a label is used, but only
// once, no matter how often it is used
class LabelCosts : public DiscreteEnergyFunctionBase
{
  public:
    inline static std::string serialization_key()
    {
        return "label-costs";
    }

    using base_type = DiscreteEnergyFunctionBase;
    using base_type::value;

    LabelCosts() = default;

    inline LabelCosts(std::size_t arity, std::initializer_list<energy_type> costs)
        : arity_(arity),
          costs_(costs),
          is_used_(costs_.size(), 0)
    {
    }

    template <typename ITER>
    LabelCosts(std::size_t arity, ITER begin, ITER end)
        : arity_(arity),
          costs_(begin, end),
          is_used_(costs_.size(), 0)
    {
    }

    discrete_label_type shape(std::size_t index) const override;

    std::size_t arity() const override;

    std::size_t size() const override;

    energy_type value(const discrete_label_type *discrete_labels) const override;

    std::unique_ptr<DiscreteEnergyFunctionBase> clone() const override;
    void add_to_lp(IlpData &ilp_data, const std::size_t *indicator_variables_mapping) const override;

    static std::unique_ptr<DiscreteEnergyFunctionBase> deserialize_json(const nlohmann::json &json);
    static std::unique_ptr<DiscreteEnergyFunctionBase> deserialize(Deserializer &deserializer);
    nlohmann::json serialize_json() const override;
    void serialize(Serializer &serializer) const override;

  private:
    std::size_t arity_;
    std::vector<energy_type> costs_;
    mutable std::vector<std::uint8_t> is_used_;

#ifndef NXTGM_NO_THREADS
    mutable std::mutex mtx_;
#endif
};

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
