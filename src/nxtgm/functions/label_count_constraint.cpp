#include <iomanip>
#include <nxtgm/functions/label_count_constraint.hpp>
#include <nxtgm/utils/n_nested_loops.hpp>

namespace nxtgm
{

LabelCountConstraint::LabelCountConstraint(std::size_t arity, const std::vector<std::size_t> &min_counts,
                                           const std::vector<std::size_t> &max_counts, energy_type scale)
    : arity_(arity),
      min_counts_(min_counts),
      max_counts_(max_counts),
      scale_(scale),
      counts_(arity)
#ifndef NXTGM_NO_THREADS
      ,
      mtx_()
#endif
{
    if (arity < 2)
    {
        throw std::runtime_error("LabelCountConstraint arity must be >= 2");
    }
    if (min_counts_.size() != max_counts_.size())
    {
        throw std::runtime_error("LabelCountConstraint min_counts and max_counts must have the same size");
    }
}

std::size_t LabelCountConstraint::arity() const
{
    return arity_;
}

discrete_label_type LabelCountConstraint::num_labels() const
{
    return min_counts_.size();
}

energy_type LabelCountConstraint::value(const discrete_label_type *discrete_labels) const
{
#ifndef NXTGM_NO_THREADS
    std::lock_guard<std::mutex> lck(mtx_);
#endif

    std::fill(counts_.begin(), counts_.end(), 0);

    auto n_duplicates = 0;

    for (std::size_t i = 0; i < arity_; ++i)
    {
        counts_[discrete_labels[i]] += 1;
    }
    energy_type how_violated = 0;
    for (std::size_t i = 0; i < counts_.size(); ++i)
    {
        if (counts_[i] < min_counts_[i])
        {
            how_violated += min_counts_[i] - counts_[i];
        }
        if (counts_[i] > max_counts_[i])
        {
            how_violated += counts_[i] - max_counts_[i];
        }
    }
    return how_violated * scale_;
}

std::unique_ptr<DiscreteConstraintFunctionBase> LabelCountConstraint::clone() const
{
    return std::make_unique<LabelCountConstraint>(arity_, min_counts_, max_counts_, scale_);
}

nlohmann::json LabelCountConstraint::serialize_json() const
{
    return {{"type", LabelCountConstraint::serialization_key()},
            {"arity", arity_},
            {"min_counts", min_counts_},
            {"max_counts", max_counts_},
            {"scale", scale_}};
}

void LabelCountConstraint::serialize(Serializer &serializer) const
{
    serializer(LabelCountConstraint::serialization_key());
    serializer(arity_);
    serializer(min_counts_);
    serializer(max_counts_);
    serializer(scale_);
}

std::unique_ptr<DiscreteConstraintFunctionBase> LabelCountConstraint::deserialize(Deserializer &deserializer)
{
    std::size_t arity;
    std::vector<std::size_t> min_counts;
    std::vector<std::size_t> max_counts;
    energy_type scale;

    deserializer(arity);
    deserializer(min_counts);
    deserializer(max_counts);
    deserializer(scale);

    return std::make_unique<LabelCountConstraint>(arity, min_counts, max_counts, scale);
}

std::unique_ptr<DiscreteConstraintFunctionBase> LabelCountConstraint::deserialize_json(const nlohmann::json &json)
{
    return std::make_unique<LabelCountConstraint>(
        json["arity"].get<std::size_t>(), json["min_counts"].get<std::vector<std::size_t>>(),
        json["max_counts"].get<std::vector<std::size_t>>(), json["scale"].get<energy_type>());
}

std::size_t LabelCountConstraint::min_counts(discrete_label_type l) const
{
    return min_counts_[l];
}
std::size_t LabelCountConstraint::max_counts(discrete_label_type l) const
{
    return max_counts_[l];
}

} // namespace nxtgm
