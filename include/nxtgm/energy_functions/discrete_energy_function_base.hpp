#pragma once

#include <initializer_list>

#include <nlohmann/json.hpp>
#include <nxtgm/nxtgm.hpp>
#include <nxtgm/utils/lp.hpp>

namespace nxtgm
{

class DiscreteEnergyFunctionBase;

// when we build an ilp to solve a discrete gm, for each function (ie the values
// of a factor) we call function->add_to_lp(...) This way, functions classes can
// specialze the way they are added to the lp. For instance the "LabelCost" has
// a effiecient way to be added to the lp. But "add_to_lp" might need to
// allocate memory, in particular for getting the energies / values of the
// function.  Therefore we use a multi purpose buffer to avoid allocating memory
// for each function.
struct IlpFactorBuilderBuffer
{
    void ensure_size(std::size_t max_factor_size, std::size_t max_factor_arity);
    std::vector<energy_type> energy_buffer;
    std::vector<discrete_label_type> label_buffer;
    std::vector<discrete_label_type> shape_buffer;
};

class DiscreteEnergyFunctionBase
{
  public:
    virtual ~DiscreteEnergyFunctionBase() = default;

    // pure virtual functions:
    virtual std::size_t arity() const = 0;
    virtual discrete_label_type shape(std::size_t index) const = 0;
    virtual energy_type energy(const discrete_label_type *discrete_labels) const = 0;

    // function with default implementation:
    virtual std::size_t size() const;
    virtual energy_type energy(std::initializer_list<discrete_label_type> discrete_labels) const;

    virtual void copy_energies(energy_type *energies) const;
    virtual void copy_shape(discrete_label_type *shape) const;

    virtual void add_energies(energy_type *energies) const;

    virtual void add_to_lp(IlpData &ilp_data, const std::size_t *indicator_variables_mapping) const;

    virtual void compute_factor_to_variable_messages(const energy_type *const *in_messages,
                                                     energy_type **out_messages) const;

    virtual std::unique_ptr<DiscreteEnergyFunctionBase> clone() const = 0;

    virtual std::unique_ptr<DiscreteEnergyFunctionBase> bind(span<const std::size_t> binded_vars,
                                                             span<const discrete_label_type> binded_vars_labels) const;

    virtual nlohmann::json serialize_json() const = 0;
};

// helper class to have a shape object
// with operator[] and size()
class DiscreteEnergyFunctionShape
{
  public:
    inline DiscreteEnergyFunctionShape(const DiscreteEnergyFunctionBase *function)
        : function_(function)
    {
    }

    inline std::size_t size() const
    {
        return function_->arity();
    }
    inline discrete_label_type operator[](std::size_t index) const
    {
        return function_->shape(index);
    }

  private:
    const DiscreteEnergyFunctionBase *function_;
};

using DiscretEnergyFunctionSerializationFactory =
    std::unordered_map<std::string, std::function<std::unique_ptr<DiscreteEnergyFunctionBase>(const nlohmann::json &)>>;

std::unique_ptr<DiscreteEnergyFunctionBase> discrete_energy_function_deserialize_json(
    const nlohmann::json &json,
    const DiscretEnergyFunctionSerializationFactory &user_factory = DiscretEnergyFunctionSerializationFactory());
} // namespace nxtgm
