#include <nxtgm/functions/label_count_constraint_base.hpp>

namespace nxtgm
{

void LabelCountConstraintBase::add_to_lp(IlpData &ilp_data, const std::size_t *indicator_variables_mapping) const
{
    const auto arity = this->arity();
    const auto num_labels = this->num_labels();

    for (discrete_label_type l = 0; l < num_labels; ++l)
    {
        // is constraint needed for that label?
        const auto min_count = this->min_counts(l);
        const auto max_count = this->max_counts(l);

        if (min_count == 0 && max_count >= arity)
        {
            continue;
        }

        ilp_data.begin_constraint(min_count, max_count);
        for (auto i = 0; i < arity; ++i)
        {
            ilp_data.add_constraint_coefficient(indicator_variables_mapping[i] + l, 1.0);
        }
    }
}

discrete_label_type LabelCountConstraintBase::shape(std::size_t) const
{
    return this->num_labels();
}

std::size_t LabelCountConstraintBase::size() const
{
    return std::pow(this->num_labels(), this->arity());
}

} // namespace nxtgm
