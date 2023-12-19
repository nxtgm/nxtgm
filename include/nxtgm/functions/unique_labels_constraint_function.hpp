#pragma once

#include <nxtgm/functions/label_count_constraint_base.hpp>
#include <nxtgm/nxtgm.hpp>

#include <xtensor/xarray.hpp>

namespace nxtgm
{

class UniqueLables : public LabelCountConstraintBase
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
                 discrete_label_type ignore_label = 0, energy_type scale = 1.0);

    std::size_t arity() const override;
    discrete_label_type num_labels() const override;
    energy_type value(const discrete_label_type *discrete_labels) const override;

    std::unique_ptr<DiscreteConstraintFunctionBase> clone() const override;
    void add_to_lp(IlpData &ilp_data, const std::size_t *indicator_variables_mapping) const override;

    nlohmann::json serialize_json() const override;
    void serialize(Serializer &serializer) const override;
    static std::unique_ptr<DiscreteConstraintFunctionBase> deserialize(Deserializer &deserializer);
    static std::unique_ptr<DiscreteConstraintFunctionBase> deserialize_json(const nlohmann::json &json);

    std::size_t min_counts(discrete_label_type l) const override;
    std::size_t max_counts(discrete_label_type l) const override;
    bool with_ignore_label() const;
    discrete_label_type ignore_label() const;

    void fuse(const discrete_label_type *labels_a, const discrete_label_type *labels_b, discrete_label_type *labels_ab,
              const std::size_t fused_arity, const std::size_t *fuse_factor_var_pos, Fusion &fusion) const override;

    void compute_to_variable_messages(const energy_type *const *in_messages, energy_type **out_messages,
                                      energy_type constraint_scaling_factor,
                                      const OptimizerParameters &optimizer_parameters) const override;

  private:
    std::size_t arity_;
    discrete_label_type n_labels_;
    bool with_ignore_label_;
    discrete_label_type ignore_label_;
    energy_type scale_;
};

} // namespace nxtgm
