#include <algorithm>
#include <cmath>
#include <nxtgm/functions/sparse_energy_function.hpp>

namespace nxtgm
{
std::size_t SparseDiscreteEnergyFunction::arity() const
{
    return data_.dimension();
}
discrete_label_type SparseDiscreteEnergyFunction::shape(std::size_t i) const
{
    return data_.shape(i);
}
std::size_t SparseDiscreteEnergyFunction::size() const
{
    return data_.size();
}
energy_type SparseDiscreteEnergyFunction::value(const discrete_label_type *discrete_labels) const
{
    return data_[discrete_labels];
}

std::unique_ptr<DiscreteEnergyFunctionBase> SparseDiscreteEnergyFunction::clone() const
{
    auto uptr = std::make_unique<SparseDiscreteEnergyFunction>(this->data_.shape());
    uptr.get()->data_.non_zero_entries() = this->data_.non_zero_entries();
    return uptr;
}

void SparseDiscreteEnergyFunction::copy_values(energy_type *energies) const
{
    std::fill(energies, energies + data_.size(), 0);

    for (const auto &item : data_.non_zero_entries())
    {
        energies[item.first] = item.second;
    }
}
void SparseDiscreteEnergyFunction::add_values(energy_type *energies) const
{
    for (const auto &item : data_.non_zero_entries())
    {
        energies[item.first] += item.second;
    }
}

std::unique_ptr<DiscreteEnergyFunctionBase> SparseDiscreteEnergyFunction::deserialize_json(const nlohmann::json &json)
{
    std::vector<discrete_label_type> shape = json["shape"];
    std::unordered_map<std::size_t, energy_type> non_zero_entries = json["non_zero_entries"];

    auto f = std::make_unique<SparseDiscreteEnergyFunction>(shape);
    f.get()->data_.non_zero_entries() = non_zero_entries;
    return f;
}

std::unique_ptr<DiscreteEnergyFunctionBase> SparseDiscreteEnergyFunction::deserialize(Deserializer &deserializer)
{
    std::vector<std::size_t> shape;
    deserializer(shape);
    auto f = std::make_unique<SparseDiscreteEnergyFunction>(shape);
    deserializer(f.get()->data_.non_zero_entries());
    return f;
}

void SparseDiscreteEnergyFunction::serialize(Serializer &serializer) const
{
    serializer(SparseDiscreteEnergyFunction::serialization_key());
    serializer(data_.shape());
    serializer(data_.non_zero_entries());
}

nlohmann::json SparseDiscreteEnergyFunction::serialize_json() const
{
    return {{"type", SparseDiscreteEnergyFunction::serialization_key()},
            {"shape", this->data_.shape()},
            {"non_zero_entries", this->data_.non_zero_entries()}};
}

void SparseDiscreteEnergyFunction::add_to_lp(IlpData &ilp_data, const std::size_t *indicator_variables_mapping) const
{
    const auto arity = this->arity();
    if (arity == 1)
    {
        for (const auto &[label, energy] : data_.non_zero_entries())
        {
            ilp_data.add_objective(indicator_variables_mapping[0] + label, energy);
        }
    }
    else
    {
        small_arity_vector<discrete_label_type> labels(arity);
        // for each entry in energies_
        for (const auto &[flat_index, energy] : data_.non_zero_entries())
        {
            // convert flat index to discrete labels
            data_.multindex_from_flat_index(flat_index, labels.data());

            // add indicator variable
            auto indicator_var =
                /* it MUST be an integer variable */
                ilp_data.add_variable(0.0, 1.0, double(energy), true);

            ilp_data.begin_constraint(0.0, double(arity) - 1);
            ilp_data.add_constraint_coefficient(indicator_var, -1.0 * arity);

            // add constraint
            for (std::size_t i = 0; i < arity; ++i)
            {
                ilp_data.add_constraint_coefficient(indicator_variables_mapping[i] + labels[i], 1.0);
            }
        }
    }
}

void SparseDiscreteEnergyFunction::set_energy(std::initializer_list<discrete_label_type> labels, energy_type energy)
{
    set_energy(labels.begin(), energy);
}

void SparseDiscreteEnergyFunction::set_energy(const discrete_label_type *labels, energy_type energy)
{
    data_[labels] = energy;
}

} // namespace nxtgm
