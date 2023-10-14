#include <algorithm>
#include <iostream>
#include <nxtgm/nxtgm.hpp>
#include <nxtgm/spaces/discrete_space.hpp>

namespace nxtgm
{
discrete_label_type DiscreteSpace::max_num_labels() const
{
    // get max element
    return *std::max_element(n_labels_.begin(), n_labels_.end());
}

nlohmann::json DiscreteSpace::serialize_json() const
{
    if (this->is_simple())
    {
        return {{"is_simple", true}, {"n_variables", n_variables_}, {"n_labels", n_labels_.front()}};
    }
    else
    {
        return {{"is_simple", false}, {"n_variables", n_variables_}, {"n_labels", n_labels_}};
    }
}

DiscreteSpace DiscreteSpace::deserialize_json(const nlohmann::json &json)
{
    if (json["is_simple"].get<bool>())
    {
        return DiscreteSpace(json["n_variables"].get<std::size_t>(), json["n_labels"].get<std::size_t>());
    }
    else
    {
        return DiscreteSpace(json["n_labels"].get<std::vector<discrete_label_type>>());
    }
}

// when is_include_mask is true,
// the subspace is the one that is true in the mask
// when is_include_mask is false,
// the subspace is the one that is false in the mask
std::pair<DiscreteSpace, std::unordered_map<std::size_t, std::size_t>> DiscreteSpace::subspace(
    span<const std::uint8_t> mask, bool is_include_mask) const
{
    std::unordered_map<std::size_t, std::size_t> space_to_subspace;
    std::size_t sub_space_vi = 0;

    for (std::size_t space_vi = 0; space_vi < n_variables_; ++space_vi)
    {
        if (static_cast<bool>(mask[space_vi]) == is_include_mask)
        {
            space_to_subspace[space_vi] = sub_space_vi;
            ++sub_space_vi;
        }
    }
    const auto num_subspace_variables = sub_space_vi;
    if (is_simple_)
    {
        return std::make_pair(DiscreteSpace(num_subspace_variables, n_labels_.front()), space_to_subspace);
    }
    else
    {
        std::vector<discrete_label_type> new_n_labels(num_subspace_variables);
        sub_space_vi = 0;
        for (std::size_t space_vi = 0; space_vi < n_variables_; ++space_vi)
        {
            if (static_cast<bool>(mask[space_vi]) == is_include_mask)
            {
                new_n_labels[sub_space_vi] = n_labels_[space_vi];
                ++sub_space_vi;
            }
        }
        return std::make_pair(DiscreteSpace(new_n_labels), space_to_subspace);
    }
}

void DiscreteSpace::serialize(Serializer &serializer) const
{
    serializer(is_simple_);
    serializer(n_variables_);
    serializer(n_labels_);
}
DiscreteSpace DiscreteSpace::deserialize(Deserializer &deserializer)
{
    DiscreteSpace space;
    deserializer(space.is_simple_);
    deserializer(space.n_variables_);
    deserializer(space.n_labels_);
    return space;
}

IndicatorVariableMapping::IndicatorVariableMapping(const DiscreteSpace &space)
    : space_(space),
      mapping_(space.is_simple() ? 1 : space.size())
{
    if (space_.is_simple())
    {
        mapping_[0] = space[0];
        n_variables_ = space.size() * space[0];
    }
    else
    {
        n_variables_ = 0;
        for (std::size_t vi = 0; vi < space.size(); ++vi)
        {
            mapping_[vi] = n_variables_;
            n_variables_ += space[vi];
        }
    }
}

std::size_t IndicatorVariableMapping::operator[](std::size_t variable) const
{
    // when simple, mapping_[0] is the number of labels
    return (space_.is_simple() ? variable * mapping_[0] : mapping_[variable]);
}

} // namespace nxtgm
