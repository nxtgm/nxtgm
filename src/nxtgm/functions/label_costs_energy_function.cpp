#include <algorithm>
#include <cmath>
#include <nxtgm/functions/label_costs_energy_function.hpp>

namespace nxtgm
{
// find the arg minimum and second arg minimum value in a range
inline std::pair<std::size_t, std::size_t> arg2min(const energy_type *begin, const energy_type *end)
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

discrete_label_type LabelCosts::shape(std::size_t index) const
{
    return costs_.size();
}

std::size_t LabelCosts::arity() const
{
    return arity_;
}

std::size_t LabelCosts::size() const
{
    return std::pow(costs_.size(), arity_);
}

energy_type LabelCosts::value(const discrete_label_type *discrete_labels) const
{

#ifndef NXTGM_NO_THREADS
    std::lock_guard<std::mutex> lck(mtx_);
#endif

    std::fill(is_used_.begin(), is_used_.end(), 0);
    for (std::size_t i = 0; i < arity_; ++i)
    {
        is_used_[discrete_labels[i]] = 1;
    }
    energy_type result = 0;
    for (std::size_t i = 0; i < is_used_.size(); ++i)
    {
        result += is_used_[i] * costs_[i];
    }
    return result;
}

std::unique_ptr<DiscreteEnergyFunctionBase> LabelCosts::clone() const
{
    return std::make_unique<LabelCosts>(arity_, costs_.begin(), costs_.end());
}

void LabelCosts::add_to_lp(IlpData &ilp_data, const std::size_t *indicator_variables_mapping) const
{
    const auto label_indicator_variables_begin = ilp_data.num_variables();

    // add n_labels varialbes
    ilp_data.add_variables(0, 1, costs_.begin(), costs_.end(), false);

    for (std::size_t ai = 0; ai < arity_; ++ai)
    {

        for (discrete_label_type l = 0; l < static_cast<discrete_label_type>(costs_.size()); ++l)
        {

            ilp_data.begin_constraint(0.0, 1.0);
            ilp_data.add_constraint_coefficient(label_indicator_variables_begin + l, 1.0);
            ilp_data.add_constraint_coefficient(indicator_variables_mapping[ai] + l, -1.0);
        }
    }

    for (discrete_label_type l = 0; l < static_cast<discrete_label_type>(costs_.size()); ++l)
    {
        ilp_data.begin_constraint(-1.0 * arity_, 0);
        ilp_data.add_constraint_coefficient(label_indicator_variables_begin + l, 1.0);

        for (std::size_t ai = 0; ai < arity_; ++ai)
        {
            ilp_data.add_constraint_coefficient(indicator_variables_mapping[ai] + l, -1.0);
        }
    }
}

nlohmann::json LabelCosts::serialize_json() const
{
    return {{"type", LabelCosts::serialization_key()}, {"arity", arity_}, {"values", costs_}};
}

void LabelCosts::serialize(Serializer &serializer) const
{
    serializer(LabelCosts::serialization_key());
    serializer(arity_);
    serializer(costs_);
}
std::unique_ptr<DiscreteEnergyFunctionBase> LabelCosts::deserialize(Deserializer &deserializer)
{
    auto f = new LabelCosts();
    deserializer(f->arity_);
    deserializer(f->costs_);
    return std::unique_ptr<DiscreteEnergyFunctionBase>(f);
}

std::unique_ptr<DiscreteEnergyFunctionBase> LabelCosts::deserialize_json(const nlohmann::json &json)
{
    return std::make_unique<LabelCosts>(json["arity"], json["values"].begin(), json["values"].end());
}

} // namespace nxtgm
