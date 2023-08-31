#include <nxtgm/constraint_functions/discrete_constraint_function_base.hpp>
#include <nxtgm/constraint_functions/discrete_constraints.hpp>
#include <nxtgm/utils/tuple_for_each.hpp>

namespace nxtgm
{

void IlpConstraintBuilderBuffer::ensure_size(std::size_t max_constraint_size, std::size_t max_constraint_arity)
{
    if (how_violated_buffer.size() < max_constraint_size)
    {
        how_violated_buffer.resize(max_constraint_size * 2);
    }
    if (label_buffer.size() < max_constraint_arity)
    {
        label_buffer.resize(max_constraint_arity * 2);
    }
    if (shape_buffer.size() < max_constraint_arity)
    {
        shape_buffer.resize(max_constraint_arity * 2);
    }
}

std::size_t DiscreteConstraintFunctionBase::size() const
{
    std::size_t result = 1;
    for (std::size_t i = 0; i < arity(); ++i)
    {
        result *= shape(i);
    }
    return result;
}
energy_type DiscreteConstraintFunctionBase::how_violated(std::initializer_list<discrete_label_type> labels) const
{
    return this->how_violated(labels.begin());
}
void DiscreteConstraintFunctionBase::add_to_lp(IlpData &ilp_data, const std::size_t *indicator_variables_mapping) const
{
    throw std::runtime_error("Not implemented");
}

std::unique_ptr<DiscreteConstraintFunctionBase> DiscreteConstraintFunctionBase::bind(
    const span<std::size_t> &binded_vars, const span<discrete_label_type> &binded_vars_labels) const
{
    throw std::runtime_error("DiscreteConstraintFunctionBase::bind is not implemented");
}

template <class T>
struct Identity
{
    using type = T;
};

using AllInternalDiscreteConstraintFunctionTypes =
    std::tuple<Identity<UniqueLables>, Identity<ArrayDiscreteConstraintFunction>>;

// yes, this if/else for each function is
// a tight coupling between the serialization and the
// concrete function types.
// A "more generic" solution would be to have a
// singleton with a map from type to factory function
// but this makes linkage more complicated
std::unique_ptr<DiscreteConstraintFunctionBase> discrete_constraint_function_deserialize_json(
    const nlohmann::json &json, const DiscretConstraintFunctionSerializationFactory &user_factory)
{
    const std::string type = json.at("type").get<std::string>();
    std::unique_ptr<DiscreteConstraintFunctionBase> result;

    AllInternalDiscreteConstraintFunctionTypes all_types;
    tuple_breakable_for_each(all_types, [&](auto &&tuple_element) {
        using function_type = typename std::decay_t<decltype(tuple_element)>::type;
        const std::string name = function_type::serialization_key();
        if (name == type)
        {
            result = std::move(function_type::deserialize_json(json));
            return false;
        }
        return true;
    });
    if (result)
    {
        return std::move(result);
    }
    else
    {
        // check if in user factory
        auto factory = user_factory.find(type);
        if (factory == user_factory.end())
        {
            throw std::runtime_error("Unknown type: `" + type + "`");
        }
        else
        {
            return factory->second(json);
        }
    }
}

} // namespace nxtgm
