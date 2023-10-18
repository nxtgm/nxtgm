#pragma once

#include <nxtgm/functions/discrete_energy_function_base.hpp>
#include <nxtgm/nxtgm.hpp>
#include <vector>

#ifndef NXTGM_NO_THREADS
#include <mutex> // std::mutex
#endif

namespace nxtgm
{

// label costs
// pay a cost once a label is used, but only
// once, no matter how often it is used
class LabelCosts : public DiscreteEnergyFunctionBase
{
  public:
    inline static std::string serialization_key()
    {
        return "label-costs";
    }

    using base_type = DiscreteEnergyFunctionBase;
    using base_type::value;

    LabelCosts() = default;

    inline LabelCosts(std::size_t arity, std::initializer_list<energy_type> costs)
        : arity_(arity),
          costs_(costs),
          is_used_(costs_.size(), 0)
    {
    }

    template <typename ITER>
    LabelCosts(std::size_t arity, ITER begin, ITER end)
        : arity_(arity),
          costs_(begin, end),
          is_used_(costs_.size(), 0)
    {
    }

    discrete_label_type shape(std::size_t index) const override;

    std::size_t arity() const override;

    std::size_t size() const override;

    energy_type value(const discrete_label_type *discrete_labels) const override;

    std::unique_ptr<DiscreteEnergyFunctionBase> clone() const override;
    void add_to_lp(IlpData &ilp_data, const std::size_t *indicator_variables_mapping) const override;

    static std::unique_ptr<DiscreteEnergyFunctionBase> deserialize_json(const nlohmann::json &json);
    static std::unique_ptr<DiscreteEnergyFunctionBase> deserialize(Deserializer &deserializer);
    nlohmann::json serialize_json() const override;
    void serialize(Serializer &serializer) const override;

  private:
    std::size_t arity_;
    std::vector<energy_type> costs_;
    mutable std::vector<std::uint8_t> is_used_;

#ifndef NXTGM_NO_THREADS
    mutable std::mutex mtx_;
#endif
};

} // namespace nxtgm
