#pragma once

#include <nxtgm/constraint_functions/discrete_constraint_function_base.hpp>
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
    using DiscreteConstraintFunctionBase::how_violated;
    virtual ~UniqueLables()
    {
    }

    UniqueLables(std::size_t arity, discrete_label_type n_labels, energy_type scale = 1.0);

    std::size_t arity() const override;
    discrete_label_type shape(std::size_t) const override;
    std::size_t size() const override;
    energy_type how_violated(const discrete_label_type *discrete_labels) const override;

    std::unique_ptr<DiscreteConstraintFunctionBase> clone() const override;
    void add_to_lp(IlpData &ilp_data, const std::size_t *indicator_variables_mapping) const override;

    nlohmann::json serialize_json() const override;

    static std::unique_ptr<DiscreteConstraintFunctionBase> deserialize_json(const nlohmann::json &json);

  private:
    std::size_t arity_;
    discrete_label_type n_labels_;
    energy_type scale_;
};

class ArrayDiscreteConstraintFunction : public DiscreteConstraintFunctionBase
{
  public:
    inline static std::string serialization_key()
    {
        return "array";
    }

    using DiscreteConstraintFunctionBase::how_violated;

    virtual ~ArrayDiscreteConstraintFunction()
    {
    }

    template <class ARRAY>
    ArrayDiscreteConstraintFunction(ARRAY &&hw) : how_violated_(hw)
    {
    }

    std::size_t arity() const override;
    discrete_label_type shape(std::size_t) const override;
    std::size_t size() const override;

    energy_type how_violated(const discrete_label_type *discrete_labels) const override;
    std::unique_ptr<DiscreteConstraintFunctionBase> clone() const override;
    void add_to_lp(IlpData &, const std::size_t *) const override;
    nlohmann::json serialize_json() const override;
    static std::unique_ptr<DiscreteConstraintFunctionBase> deserialize_json(const nlohmann::json &json);

  private:
    xt::xarray<energy_type> how_violated_;
};

} // namespace nxtgm
