#include <nxtgm/functions/array_constraint_function.hpp>
#include <nxtgm/functions/discrete_constraint_function_base.hpp>
#include <nxtgm/functions/unique_labels_constraint_function.hpp>
#include <nxtgm/optimizers/gm/discrete/fusion.hpp>
#include <nxtgm/utils/n_nested_loops.hpp>
#include <nxtgm/utils/tuple_for_each.hpp>

namespace nxtgm
{

void DiscreteConstraintFunctionBase::add_to_lp(IlpData &ilp_data, const std::size_t *indicator_variables_mapping) const
{
    throw std::runtime_error("Not implemented");
}

std::unique_ptr<DiscreteConstraintFunctionBase> DiscreteConstraintFunctionBase::bind(
    const span<std::size_t> &binded_vars, const span<discrete_label_type> &binded_vars_labels) const
{
    throw std::runtime_error("DiscreteConstraintFunctionBase::bind is not implemented");
}

void DiscreteConstraintFunctionBase::compute_to_variable_messages(const energy_type *const *in_messages,
                                                                  energy_type **out_messages,
                                                                  energy_type constraint_scaling_factor) const
{

    const auto arity = this->arity();

    if (arity == 1)
    {
        // in case of a unary, the outmessage is just
        // the energy of the unary
        this->copy_values(out_messages[0]);
    }
    else
    {
        small_arity_vector<discrete_label_type> labels(arity);
        DiscreteFunctionShape shape(this);

        // initialize
        for (std::size_t ai = 0; ai < arity; ++ai)
        {
            std::fill(out_messages[ai], out_messages[ai] + shape[ai], std::numeric_limits<energy_type>::infinity());
        }

        n_nested_loops<discrete_label_type>(arity, shape, labels, [&](auto &&_) {
            // compute the how_violated the constraint is
            // with the current label
            const auto energy = this->value(labels.data()) * constraint_scaling_factor;

            // sum the incoming messages for that labels / config
            energy_type sum_of_incoming_messages = 0.0;
            for (std::size_t ai = 0; ai < arity; ++ai)
            {
                sum_of_incoming_messages += in_messages[ai][labels[ai]];
            }

            // fill outgoing messages
            for (std::size_t ai = 0; ai < arity; ++ai)
            {
                const auto label = labels[ai];
                const auto current_min = out_messages[ai][label];
                out_messages[ai][label] =
                    std::min(current_min, energy + (sum_of_incoming_messages - in_messages[ai][label]));
            }
        });
    }
}

void DiscreteConstraintFunctionBase::fuse(const discrete_label_type *labels_a, const discrete_label_type *labels_b,
                                          discrete_label_type *labels_ab, const std::size_t fused_arity,
                                          const std::size_t *fuse_factor_var_pos, Fusion &fusion) const
{
    // std::cout<<"DiscreteConstraintFunctionBase::fuse"<<std::endl;
    std::vector<std::size_t> shape(fused_arity, 2);
    auto fused_function_data = xt::xarray<energy_type>::from_shape(shape);
    small_arity_vector<discrete_label_type> fused_coords(fused_arity);

    // iterate over all states of the fused function
    std::size_t sub_factor_flat_index = 0;
    n_nested_loops_binary_shape(fused_arity, fused_coords.data(), [&](auto &&_) {
        for (std::size_t fi = 0; fi < fused_arity; ++fi)
        {
            const auto pos = fuse_factor_var_pos[fi];
            labels_ab[pos] = fused_coords[fi] == 0 ? labels_a[pos] : labels_b[pos];
        }

        auto e = this->value(labels_ab);
        fused_function_data[sub_factor_flat_index] = e;
        ++sub_factor_flat_index;
    });

    fusion.add_to_fuse_gm(std::make_unique<ArrayDiscreteConstraintFunction>(std::move(fused_function_data)),
                          fuse_factor_var_pos);
    // std::cout<<"DiscreteConstraintFunctionBase::fuse end"<<std::endl;
}

template <class T>
struct Identity
{
    using type = T;
};

using AllInternalDiscreteConstraintFunctionTypes =
    std::tuple<Identity<UniqueLables>, Identity<ArrayDiscreteConstraintFunction>>;

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

std::unique_ptr<DiscreteConstraintFunctionBase> discrete_constraint_function_deserialize(Deserializer &deserializer)
{
    std::string type;
    deserializer(type);

    std::unique_ptr<DiscreteConstraintFunctionBase> result;

    AllInternalDiscreteConstraintFunctionTypes all_types;
    tuple_breakable_for_each(all_types, [&](auto &&tuple_element) {
        using function_type = typename std::decay_t<decltype(tuple_element)>::type;
        const std::string name = function_type::serialization_key();
        if (name == type)
        {
            result = std::move(function_type::deserialize(deserializer));
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
        throw std::runtime_error("Unknown type: `" + type + "`");
    }
}

} // namespace nxtgm
