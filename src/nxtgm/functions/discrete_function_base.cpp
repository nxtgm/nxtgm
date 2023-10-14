#include <nxtgm/functions/discrete_function_base.hpp>
#include <nxtgm/nxtgm.hpp>
#include <nxtgm/utils/n_nested_loops.hpp>
#include <nxtgm/utils/tuple_for_each.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

namespace nxtgm
{

std::size_t DiscreteFunctionBase::size() const
{
    std::size_t result = 1;
    for (std::size_t i = 0; i < arity(); ++i)
    {
        result *= static_cast<std::size_t>(shape(i));
    }
    return result;
}

energy_type DiscreteFunctionBase::value(std::initializer_list<discrete_label_type> discrete_labels) const
{
    return this->value(discrete_labels.begin());
}

void DiscreteFunctionBase::copy_values(energy_type *energies) const
{

    const auto arity = this->arity();
    small_arity_vector<discrete_label_type> discrete_labels_buffer(arity);

    auto flat_index = 0;
    n_nested_loops<discrete_label_type>(arity, DiscreteFunctionShape(this), discrete_labels_buffer, [&](auto &&_) {
        energies[flat_index] = this->value(discrete_labels_buffer.data());
        ++flat_index;
    });
}

void DiscreteFunctionBase::add_values(energy_type *energies) const
{
    const auto arity = this->arity();

    small_arity_vector<discrete_label_type> labels(arity);

    auto flat_index = 0;
    n_nested_loops<discrete_label_type>(arity, DiscreteFunctionShape(this), labels, [&](auto &&_) {
        energies[flat_index] += this->value(labels.data());
        ++flat_index;
    });
}

void DiscreteFunctionBase::copy_shape(discrete_label_type *shape) const
{
    for (std::size_t i = 0; i < arity(); ++i)
    {
        shape[i] = this->shape(i);
    }
}

} // namespace nxtgm
