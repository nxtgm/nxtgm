#pragma once

#include <nxtgm/functions/label_count_constraint_base.hpp>
#include <nxtgm/nxtgm.hpp>
#include <vector>
#include <xtensor/xarray.hpp>

#ifndef NXTGM_NO_THREADS
#include <mutex>
#endif

namespace nxtgm
{

// require a minimum and maximum number how often a label can be used
class LabelCountConstraint : public LabelCountConstraintBase
{
  public:
    static std::string serialization_key()
    {
        return "label_count_constraint";
    }
    using LabelCountConstraintBase::value;
    virtual ~LabelCountConstraint()
    {
    }
    LabelCountConstraint() = default;
    LabelCountConstraint(std::size_t arity, const std::vector<std::size_t> &min_counts,
                         const std::vector<std::size_t> &max_counts, energy_type scale = 1.0);

    std::size_t arity() const override;
    discrete_label_type num_labels() const override;
    energy_type value(const discrete_label_type *discrete_labels) const override;

    std::unique_ptr<DiscreteConstraintFunctionBase> clone() const override;

    nlohmann::json serialize_json() const override;
    void serialize(Serializer &serializer) const override;
    static std::unique_ptr<DiscreteConstraintFunctionBase> deserialize(Deserializer &deserializer);
    static std::unique_ptr<DiscreteConstraintFunctionBase> deserialize_json(const nlohmann::json &json);

    std::size_t min_counts(discrete_label_type l) const override;
    std::size_t max_counts(discrete_label_type l) const override;

  private:
    std::size_t arity_;
    std::vector<std::size_t> min_counts_;
    std::vector<std::size_t> max_counts_;
    energy_type scale_;

    mutable std::vector<std::size_t> counts_;

#ifndef NXTGM_NO_THREADS
    mutable std::mutex mtx_;
#endif
};

} // namespace nxtgm
