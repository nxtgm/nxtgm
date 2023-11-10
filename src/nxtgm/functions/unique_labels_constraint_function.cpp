#include <iomanip>
#include <nxtgm/functions/unique_labels_constraint_function.hpp>
#include <nxtgm/utils/n_nested_loops.hpp>

namespace nxtgm
{

UniqueLables::UniqueLables(std::size_t arity, discrete_label_type n_labels, bool with_ignore_label, energy_type scale)
    : arity_(arity),
      n_labels_(n_labels),
      with_ignore_label_(with_ignore_label),
      scale_(scale)
{
    if (arity < 2)
    {
        throw std::runtime_error("UniqueLables arity must be >= 2");
    }
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

energy_type UniqueLables::value(const discrete_label_type *discrete_labels) const
{
    auto n_duplicates = 0;

    for (auto i = 0; i < arity_ - 1; ++i)
    {
        const auto label_i = discrete_labels[i];
        if (with_ignore_label_ && label_i == 0)
        {
            continue;
        }
        for (auto j = i + 1; j < arity_; ++j)
        {
            const auto label_j = discrete_labels[j];
            if (with_ignore_label_ && label_j == 0)
            {
                continue;
            }
            if (label_i == label_j)
            {
                ++n_duplicates;
            }
        }
    }
    return n_duplicates == 0 ? static_cast<energy_type>(0) : scale_ * n_duplicates;
}

std::unique_ptr<DiscreteConstraintFunctionBase> UniqueLables::clone() const
{
    return std::make_unique<UniqueLables>(arity_, n_labels_, with_ignore_label_, scale_);
}

void UniqueLables::add_to_lp(IlpData &ilp_data, const std::size_t *indicator_variables_mapping) const
{
    for (discrete_label_type l = (with_ignore_label_ ? 1 : 0); l < n_labels_; ++l)
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
    return {{"type", UniqueLables::serialization_key()},
            {"arity", arity_},
            {"num_labels", n_labels_},
            {"with_ignore_label", with_ignore_label_},
            {"scale", scale_}};
}

void UniqueLables::serialize(Serializer &serializer) const
{
    serializer(UniqueLables::serialization_key());
    serializer(arity_);
    serializer(n_labels_);
    serializer(with_ignore_label_);
    serializer(scale_);
}

std::unique_ptr<DiscreteConstraintFunctionBase> UniqueLables::deserialize(Deserializer &deserializer)
{
    auto f = new UniqueLables();
    deserializer(f->arity_);
    deserializer(f->n_labels_);
    deserializer(f->scale_);
    deserializer(f->with_ignore_label_);
    return std::unique_ptr<DiscreteConstraintFunctionBase>(f);
}

std::unique_ptr<DiscreteConstraintFunctionBase> UniqueLables::deserialize_json(const nlohmann::json &json)
{
    return std::make_unique<UniqueLables>(json["arity"].get<std::size_t>(),
                                          json["num_labels"].get<discrete_label_type>(),
                                          json["with_ignore_label"].get<bool>(), json["scale"].get<energy_type>());
}

} // namespace nxtgm
