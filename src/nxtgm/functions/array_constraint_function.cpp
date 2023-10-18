#include <nxtgm/functions/array_constraint_function.hpp>
#include <nxtgm/utils/n_nested_loops.hpp>

namespace nxtgm
{

std::size_t ArrayDiscreteConstraintFunction::arity() const
{
    return how_violated_.dimension();
}
discrete_label_type ArrayDiscreteConstraintFunction::shape(std::size_t i) const
{
    return how_violated_.shape(i);
}
std::size_t ArrayDiscreteConstraintFunction::size() const
{
    return how_violated_.size();
}

energy_type ArrayDiscreteConstraintFunction::value(const discrete_label_type *discrete_labels) const
{
    const_discrete_label_span labels(discrete_labels, how_violated_.dimension());
    return how_violated_[labels];
}
std::unique_ptr<DiscreteConstraintFunctionBase> ArrayDiscreteConstraintFunction::clone() const
{
    return std::make_unique<ArrayDiscreteConstraintFunction>(how_violated_);
}
void ArrayDiscreteConstraintFunction::add_to_lp(IlpData &ilp_data, const std::size_t *indicator_variables_mapping) const
{
    const auto arity = this->arity();
    const auto shape = DiscreteFunctionShape(this);

    auto flat_index = 0;

    small_arity_vector<discrete_label_type> labels(arity);
    n_nested_loops<discrete_label_type>(arity, shape, labels, [&](auto &&_) {
        auto hv = how_violated_[flat_index];
        if (hv > constraint_feasiblility_limit)
        {
            ilp_data.begin_constraint(0.0, arity - 1);
            for (std::size_t i = 0; i < arity; ++i)
            {
                ilp_data.add_constraint_coefficient(indicator_variables_mapping[i] + labels[i], 1.0);
            }
        }
        ++flat_index;
    });
}

nlohmann::json ArrayDiscreteConstraintFunction::serialize_json() const
{
    nlohmann::json shape = nlohmann::json::array();
    for (auto s : how_violated_.shape())
    {
        shape.push_back(s);
    }

    // iterator pair to nlhohmann::json
    auto values = nlohmann::json::array();
    for (auto it = how_violated_.begin(); it != how_violated_.end(); ++it)
    {
        values.push_back(*it);
    }

    return {{"type", "array"}, {"dimensions", how_violated_.dimension()}, {"shape", shape}, {"values", values}};
}

void ArrayDiscreteConstraintFunction::serialize(Serializer &serializer) const
{
    serializer(ArrayDiscreteConstraintFunction::serialization_key());
    serializer(how_violated_);
}

std::unique_ptr<DiscreteConstraintFunctionBase> ArrayDiscreteConstraintFunction::deserialize(Deserializer &deserializer)
{
    auto f = new ArrayDiscreteConstraintFunction();
    deserializer(f->how_violated_);
    return std::unique_ptr<DiscreteConstraintFunctionBase>(f);
}

std::unique_ptr<DiscreteConstraintFunctionBase> ArrayDiscreteConstraintFunction::deserialize_json(
    const nlohmann::json &json)
{
    std::vector<std::size_t> shape;
    for (auto s : json["shape"])
    {
        shape.push_back(s);
    }
    xt::xarray<energy_type> array(shape);
    std::copy(json["values"].begin(), json["values"].end(), array.begin());

    return std::make_unique<ArrayDiscreteConstraintFunction>(array);
}
} // namespace nxtgm
