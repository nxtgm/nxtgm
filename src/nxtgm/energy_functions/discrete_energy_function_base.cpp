

#include <nxtgm/energy_functions/discrete_energy_function_base.hpp>
#include <nxtgm/nxtgm.hpp>
#include <nxtgm/utils/n_nested_loops.hpp>
#include <nxtgm/utils/tuple_for_each.hpp>
// xtensor view
#include <xtensor/xview.hpp>
// xtensor adapt
#include <xtensor/xadapt.hpp>

#include <nxtgm/energy_functions/discrete_energy_functions.hpp>
#include <nxtgm/utils/n_nested_loops.hpp>

// bitset
#include <bitset>

namespace nxtgm
{

template <class XTENSOR>
auto bind_at(const XTENSOR &xtensor, std::size_t axis, std::size_t value)
{
    xt::xstrided_slice_vector sv(xtensor.dimension(), xt::all());
    sv[axis] = value;
    return xt::strided_view(xtensor, sv);
}

template <class XTENSOR, class AXES, class LABELS>
auto bind_many(const XTENSOR &xtensor, AXES &&axes, LABELS &&labels)
{
    xt::xstrided_slice_vector sv(xtensor.dimension(), xt::all());

    for (std::size_t i = 0; i < axes.size(); ++i)
    {
        sv[axes[i]] = labels[i];
    }

    return xt::strided_view(xtensor, sv);
}

void IlpFactorBuilderBuffer::ensure_size(std::size_t max_factor_size, std::size_t max_factor_arity)
{
    if (energy_buffer.size() < max_factor_size)
    {
        energy_buffer.resize(max_factor_size * 2);
    }
    if (label_buffer.size() < max_factor_arity)
    {
        label_buffer.resize(max_factor_arity * 2);
    }
    if (shape_buffer.size() < max_factor_arity)
    {
        shape_buffer.resize(max_factor_arity * 2);
    }
}

energy_type DiscreteEnergyFunctionBase::energy(std::initializer_list<discrete_label_type> discrete_labels) const
{
    return this->energy(discrete_labels.begin());
}

std::size_t DiscreteEnergyFunctionBase::size() const
{
    std::size_t result = 1;
    for (std::size_t i = 0; i < arity(); ++i)
    {
        result *= static_cast<std::size_t>(shape(i));
    }
    return result;
}

void DiscreteEnergyFunctionBase::copy_energies(energy_type *energies) const
{

    const auto arity = this->arity();
    small_arity_vector<discrete_label_type> discrete_labels_buffer(arity);

    auto flat_index = 0;
    n_nested_loops<discrete_label_type>(arity, DiscreteEnergyFunctionShape(this), discrete_labels_buffer,
                                        [&](auto &&_) {
                                            energies[flat_index] = this->energy(discrete_labels_buffer.data());
                                            ++flat_index;
                                        });
}

void DiscreteEnergyFunctionBase::add_energies(energy_type *energies) const
{
    const auto arity = this->arity();

    small_arity_vector<discrete_label_type> labels(arity);

    auto flat_index = 0;
    n_nested_loops<discrete_label_type>(arity, DiscreteEnergyFunctionShape(this), labels, [&](auto &&_) {
        energies[flat_index] += this->energy(labels.data());
        ++flat_index;
    });
}

void DiscreteEnergyFunctionBase::copy_shape(discrete_label_type *shape) const
{
    for (std::size_t i = 0; i < arity(); ++i)
    {
        shape[i] = this->shape(i);
    }
}

void DiscreteEnergyFunctionBase::add_to_lp(IlpData &ilp_data, const std::size_t *indicator_variables_mapping) const
{
    const auto arity = this->arity();
    const auto factor_size = this->size();

    small_vector<energy_type, 1024> energy_buffer(factor_size);

    auto energies = energy_buffer.data();
    this->copy_energies(energy_buffer.data());

    if (arity == 1)
    {
        for (discrete_label_type label = 0; label < static_cast<discrete_label_type>(factor_size); ++label)
        {
            ilp_data[indicator_variables_mapping[0] + label] += energies[label];
        }
    }
    else
    {
        small_arity_vector<discrete_label_type> shape_buffer(arity);
        auto shape = shape_buffer.data();
        this->copy_shape(shape);

        // where to the factor indicator variables start?
        const auto start = ilp_data.num_variables();
        // todo: avoid allocation?
        auto factor_indicator_vars =
            xt::eval(xt::arange(start, start + factor_size).reshape(const_discrete_label_span(shape, arity)));

        ilp_data.add_variables(0.0, 1.0, energies, energies + factor_size, false);

        for (auto ai = 0; ai < arity; ++ai)
        {

            for (auto label = 0; label < shape[ai]; ++label)
            {
                const auto var_inidcator = indicator_variables_mapping[ai] + label;
                auto constraint_vars = bind_at(factor_indicator_vars, ai, label);

                ilp_data.begin_constraint(0.0, 0.0);
                ilp_data.add_constraint_coefficient(var_inidcator, -1.0);
                for (auto var : constraint_vars)
                {
                    ilp_data.add_constraint_coefficient(var, 1.0);
                }
            }
        }
    }
}

void DiscreteEnergyFunctionBase::compute_factor_to_variable_messages(const energy_type *const *in_messages,
                                                                     energy_type **out_messages) const
{

    const auto arity = this->arity();

    if (arity == 1)
    {
        // in case of a unary, the outmessage is just
        // the energy of the unary
        this->copy_energies(out_messages[0]);
    }
    else
    {
        small_arity_vector<discrete_label_type> labels(arity);
        DiscreteEnergyFunctionShape shape(this);

        // initialize
        for (std::size_t ai = 0; ai < arity; ++ai)
        {
            std::fill(out_messages[ai], out_messages[ai] + shape[ai], std::numeric_limits<energy_type>::infinity());
        }

        n_nested_loops<discrete_label_type>(arity, shape, labels, [&](auto &&_) {
            // compute the energy of the factor
            // with the current label
            const auto energy = this->energy(labels.data());

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

std::unique_ptr<DiscreteEnergyFunctionBase> DiscreteEnergyFunctionBase::bind(
    span<const std::size_t> binded_vars, span<const discrete_label_type> binded_vars_labels) const
{
    // copy the energy
    small_factor_size_vector<energy_type> energies(this->size());
    this->copy_energies(energies.data());

    const std::size_t arity = this->arity();

    // copy the shape (this is stupid atm)
    // xtensor adapt only takes std vector
    small_arity_vector<discrete_label_type> shape(arity);
    std::vector<std::size_t> size_t_shape(arity);
    this->copy_shape(shape.data());
    std::copy(shape.begin(), shape.end(), size_t_shape.begin());

    // create an xtensor view
    auto energy_view = xt::adapt(energies.data(), this->size(), xt::no_ownership(), size_t_shape);

    // bind the variables
    auto binded = bind_many(energy_view, binded_vars, binded_vars_labels);

    return std::make_unique<XArray>(std::move(binded));
}

template <class T>
struct Identity
{
    using type = T;
};

using AllInternalDiscreteEnergyFunctionTypes =
    std::tuple<Identity<XArray>, Identity<Potts>, Identity<LabelCosts>, Identity<SparseDiscreteEnergyFunction>>;

// this function introduces a tight coupling between
// the serialization and the concrete function types.
// A "more generic" solution would be to have a
// singleton with a map from type to factory function
// but this makes linkage more complicated
std::unique_ptr<DiscreteEnergyFunctionBase> discrete_energy_function_deserialize_json(
    const nlohmann::json &json, const DiscretEnergyFunctionSerializationFactory &user_factory)
{
    const std::string type = json.at("type").get<std::string>();
    std::unique_ptr<DiscreteEnergyFunctionBase> result;

    AllInternalDiscreteEnergyFunctionTypes all_types;
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
