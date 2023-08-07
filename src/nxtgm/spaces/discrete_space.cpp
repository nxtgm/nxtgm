#include <algorithm>
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
