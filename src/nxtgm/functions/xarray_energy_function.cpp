#include <algorithm>
#include <cmath>
#include <nxtgm/functions/xarray_energy_function.hpp>

namespace nxtgm
{

XArray::XArray(const xarray_type &values)
    : values_(values)
{
}

discrete_label_type XArray::shape(std::size_t index) const
{
    return values_.shape()[index];
}

std::size_t XArray::arity() const
{
    return values_.dimension();
}

std::size_t XArray::size() const
{
    return values_.size();
}

energy_type XArray::value(const discrete_label_type *discrete_labels) const
{
    const_discrete_label_span discrete_labels_span(discrete_labels, values_.dimension());
    return values_[discrete_labels_span];
}
std::unique_ptr<DiscreteEnergyFunctionBase> XArray::clone() const
{
    return std::make_unique<XArray>(values_);
}

void XArray::copy_values(energy_type *energies) const
{
    std::copy(values_.begin(), values_.end(), energies);
}
void XArray::add_values(energy_type *energies) const
{
    std::transform(values_.data(), values_.data() + values_.size(), energies, energies, std::plus<energy_type>());
}

nlohmann::json XArray::serialize_json() const
{

    nlohmann::json shape = nlohmann::json::array();
    for (auto s : values_.shape())
    {
        shape.push_back(s);
    }

    auto values = nlohmann::json::array();
    for (auto it = values_.begin(); it != values_.end(); ++it)
    {
        values.push_back(*it);
    }

    return {{"type", XArray::serialization_key()}, {"shape", shape}, {"values", values}};
}

void XArray::serialize(Serializer &serializer) const
{
    serializer(XArray::serialization_key());
    serializer(values_);
}
std::unique_ptr<DiscreteEnergyFunctionBase> XArray::deserialize(Deserializer &deserializer)
{
    auto f = new XArray();
    deserializer(f->values_);
    return std::unique_ptr<DiscreteEnergyFunctionBase>(f);
}

std::unique_ptr<DiscreteEnergyFunctionBase> XArray::deserialize_json(const nlohmann::json &json)
{
    std::vector<std::size_t> shape;
    for (auto s : json["shape"])
    {
        shape.push_back(s);
    }
    typename XArray::xarray_type array(shape);
    std::copy(json["values"].begin(), json["values"].end(), array.begin());
    return std::make_unique<XArray>(array);
}

} // namespace nxtgm
