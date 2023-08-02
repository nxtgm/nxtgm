#include <nxtgm/constraint_functions/discrete_constraints.hpp>
#include <nxtgm/utils/n_nested_loops.hpp>

namespace nxtgm
{

UniqueLables::UniqueLables(std::size_t arity, discrete_label_type n_labels, energy_type scale)
    : arity_(arity), n_labels_(n_labels), scale_(scale)
{
    // std::cout<<"wup"<<std::endl;
}

std::size_t UniqueLables::arity() const
{
    return arity_;
}

discrete_label_type UniqueLables::shape(std::size_t) const
{
    return n_labels_;
}

std::size_t UniqueLables::size() const
{
    return std::pow(n_labels_, arity_);
}

energy_type UniqueLables::how_violated(const discrete_label_type *discrete_labels) const
{
    auto n_duplicates = 0;

    for (auto i = 0; i < arity_ - 1; ++i)
    {
        for (auto j = i + 1; j < arity_; ++j)
        {
            if (discrete_labels[i] == discrete_labels[j])
            {
                ++n_duplicates;
            }
        }
    }
    return n_duplicates == 0 ? static_cast<energy_type>(0) : scale_ * n_duplicates;
}

std::unique_ptr<DiscreteConstraintFunctionBase> UniqueLables::clone() const
{
    return std::make_unique<UniqueLables>(arity_, n_labels_, scale_);
}

void UniqueLables::add_to_lp(IlpData &ilp_data, const std::size_t *indicator_variables_mapping) const
{
    for (discrete_label_type l = 0; l < n_labels_; ++l)
    {

        ilp_data.begin_constraint(0.0, 1.0);
        for (auto i = 0; i < arity_; ++i)
        {
            ilp_data.add_constraint_coefficient(indicator_variables_mapping[i] + l, 1.0);
        }
    }
}

nlohmann::json UniqueLables::serialize_json() const
{
    return {
        {"type", UniqueLables::serialization_key()}, {"arity", arity_}, {"num_labels", n_labels_}, {"scale", scale_}};
}

std::unique_ptr<DiscreteConstraintFunctionBase> UniqueLables::deserialize_json(const nlohmann::json &json)
{
    return std::make_unique<UniqueLables>(json["arity"].get<std::size_t>(),
                                          json["num_labels"].get<discrete_label_type>(),
                                          json["scale"].get<energy_type>());
}

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

energy_type ArrayDiscreteConstraintFunction::how_violated(const discrete_label_type *discrete_labels) const
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
    const auto shape = DiscreteConstraintFunctionShape(this);

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
