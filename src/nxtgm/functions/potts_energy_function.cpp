#include <algorithm>
#include <cmath>
#include <nxtgm/functions/potts_energy_function.hpp>
#include <nxtgm/functions/xtensor_energy_function.hpp>

namespace nxtgm
{

// find the arg minimum and second arg minimum value in a range
std::pair<std::size_t, std::size_t> arg2min(const energy_type *begin, const energy_type *end)
{
    std::pair<std::size_t, std::size_t> res;
    energy_type min0 = std::numeric_limits<energy_type>::infinity();
    energy_type min1 = min0;
    const auto dist = std::distance(begin, end);
    for (auto i = 0; i < dist; ++i)
    {
        const energy_type val = begin[i];
        if (val < min1)
        {
            if (val < min0)
            {

                min1 = min0;
                res.second = res.first;
                min0 = val;
                res.first = i;
                continue;
            }
            else
            {
                min1 = val;
                res.second = i;
            }
        }
    }
    return res;
}

Potts::Potts(std::size_t num_labels, energy_type beta)
    : num_labels_(num_labels),
      beta_(beta)
{
}

std::size_t Potts::arity() const
{
    return 2;
}

discrete_label_type Potts::shape(std::size_t) const
{
    return num_labels_;
}

std::size_t Potts::size() const
{
    return num_labels_ * num_labels_;
}

energy_type Potts::value(const discrete_label_type *discrete_labels) const
{
    return beta_ * (discrete_labels[0] != discrete_labels[1]);
}
std::unique_ptr<DiscreteEnergyFunctionBase> Potts::clone() const
{
    return std::make_unique<Potts>(num_labels_, beta_);
}

void Potts::copy_values(energy_type *energies) const
{
    for (std::size_t i = 0; i < num_labels_; ++i)
    {
        for (std::size_t j = 0; j < num_labels_; ++j)
        {
            energies[i * num_labels_ + j] = beta_ * (i != j);
        }
    }
}
void Potts::add_values(energy_type *energies) const
{
    for (std::size_t i = 0; i < num_labels_; ++i)
    {
        for (std::size_t j = 0; j < num_labels_; ++j)
        {
            if (i != j)
            {
                energies[i * num_labels_ + j] += beta_;
            }
        }
    }
}

void Potts::compute_factor_to_variable_messages(const energy_type *const *in_messages, energy_type **out_messages) const
{

    if (beta_ >= 0)
    {
        const energy_type minIn0Beta = *std::min_element(in_messages[0], in_messages[0] + num_labels_) + beta_;
        const energy_type minIn1Beta = *std::min_element(in_messages[1], in_messages[1] + num_labels_) + beta_;
        for (discrete_label_type l = 0; l < num_labels_; ++l)
        {
            out_messages[0][l] = std::min(in_messages[1][l], minIn1Beta);
            out_messages[1][l] = std::min(in_messages[0][l], minIn0Beta);
        }
    }
    else
    {
        discrete_label_type aMin0, aMin1, aSMin0, aSMin1;
        std::tie(aMin0, aSMin0) = arg2min(in_messages[0], in_messages[0] + num_labels_);
        std::tie(aMin1, aSMin1) = arg2min(in_messages[1], in_messages[1] + num_labels_);
        const energy_type min0 = in_messages[0][aMin0];
        const energy_type min1 = in_messages[1][aMin1];
        const energy_type min0Beta = min0 + beta_;
        const energy_type min1Beta = min1 + beta_;
        const energy_type smin0Beta = std::min(in_messages[0][aSMin0] + beta_, min0);
        const energy_type smin1Beta = std::min(in_messages[1][aSMin1] + beta_, min1);
        for (discrete_label_type l = 0; l < num_labels_; ++l)
        {
            out_messages[0][l] = aMin1 != l ? min1Beta : smin1Beta;
            out_messages[1][l] = aMin0 != l ? min0Beta : smin0Beta;
        }
    }
}

std::unique_ptr<DiscreteEnergyFunctionBase> Potts::bind(span<const std::size_t> binded_vars,
                                                        span<const discrete_label_type> binded_vars_labels) const
{
    auto values = xt::xtensor<energy_type, 1>::from_shape({std::size_t(num_labels_)});
    values.fill(beta_);
    values[binded_vars_labels[0]] = 0.0;
    return std::make_unique<Unary>(std::move(values));
}

nlohmann::json Potts::serialize_json() const
{
    return {{"type", Potts::serialization_key()}, {"num_labels", num_labels_}, {"beta", beta_}};
}
void Potts::serialize(Serializer &serializer) const
{
    serializer(Potts::serialization_key());
    serializer(num_labels_);
    serializer(beta_);
}

std::unique_ptr<DiscreteEnergyFunctionBase> Potts::deserialize_json(const nlohmann::json &json)
{
    return std::make_unique<Potts>(json["num_labels"], json["beta"]);
}
std::unique_ptr<DiscreteEnergyFunctionBase> Potts::deserialize(Deserializer &deserializer)
{
    auto p = new Potts();
    deserializer(p->num_labels_);
    deserializer(p->beta_);
    return std::unique_ptr<DiscreteEnergyFunctionBase>(p);
}

} // namespace nxtgm
