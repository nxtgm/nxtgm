#pragma once

#include <nxtgm/functions/discrete_constraint_function_base.hpp>
#include <nxtgm/nxtgm.hpp>

#include <xtensor/xarray.hpp>

namespace nxtgm
{

class ArrayDiscreteConstraintFunction : public DiscreteConstraintFunctionBase
{
  public:
    inline static std::string serialization_key()
    {
        return "array";
    }

    using DiscreteConstraintFunctionBase::value;

    virtual ~ArrayDiscreteConstraintFunction()
    {
    }
    ArrayDiscreteConstraintFunction() = default;
    template <class ARRAY>
    ArrayDiscreteConstraintFunction(ARRAY &&hw)
        : how_violated_(hw)
    {
    }

    std::size_t arity() const override;
    discrete_label_type shape(std::size_t) const override;
    std::size_t size() const override;

    energy_type value(const discrete_label_type *discrete_labels) const override;
    std::unique_ptr<DiscreteConstraintFunctionBase> clone() const override;
    void add_to_lp(IlpData &, const std::size_t *) const override;
    nlohmann::json serialize_json() const override;
    void serialize(Serializer &serializer) const override;

    static std::unique_ptr<DiscreteConstraintFunctionBase> deserialize(Deserializer &deserializer);
    static std::unique_ptr<DiscreteConstraintFunctionBase> deserialize_json(const nlohmann::json &json);

  private:
    xt::xarray<energy_type> how_violated_;
};

} // namespace nxtgm
