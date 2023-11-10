#pragma once

#include <nxtgm/functions/discrete_constraint_function_base.hpp>
#include <nxtgm/nxtgm.hpp>

#include <xtensor/xarray.hpp>

namespace nxtgm
{

class UniqueLables : public DiscreteConstraintFunctionBase
{
  public:
    static std::string serialization_key()
    {
        return "unique";
    }
    using DiscreteConstraintFunctionBase::value;
    virtual ~UniqueLables()
    {
    }
    UniqueLables() = default;
    UniqueLables(std::size_t arity, discrete_label_type n_labels, bool with_ignore_label = false,
                 energy_type scale = 1.0);

    std::size_t arity() const override;
    discrete_label_type shape(std::size_t) const override;
    std::size_t size() const override;
    energy_type value(const discrete_label_type *discrete_labels) const override;

    std::unique_ptr<DiscreteConstraintFunctionBase> clone() const override;
    void add_to_lp(IlpData &ilp_data, const std::size_t *indicator_variables_mapping) const override;

    nlohmann::json serialize_json() const override;
    void serialize(Serializer &serializer) const override;
    static std::unique_ptr<DiscreteConstraintFunctionBase> deserialize(Deserializer &deserializer);
    static std::unique_ptr<DiscreteConstraintFunctionBase> deserialize_json(const nlohmann::json &json);

  private:
    std::size_t arity_;
    discrete_label_type n_labels_;
    bool with_ignore_label_;
    energy_type scale_;
};

} // namespace nxtgm
