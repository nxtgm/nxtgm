#include <fstream>
#include <iostream>
#include <nxtgm/models/gm/discrete_gm.hpp>
#include <nxtgm/nxtgm.hpp>

namespace nxtgm
{
const_discrete_label_span local_solution_from_model_solution(const std::vector<std::size_t> &variables,
                                                             const span<const discrete_label_type> &solution,
                                                             std::vector<discrete_label_type> &local_labels_buffer)
{
    const auto arity = variables.size();
    if (local_labels_buffer.size() < arity)
    {
        local_labels_buffer.resize(arity * 2);
    }

    auto i = 0;
    for (auto vi : variables)
    {
        local_labels_buffer[i] = solution[vi];
        i++;
    }
    return const_discrete_label_span(local_labels_buffer.data(), arity);
}

DiscreteGm::DiscreteGm(const DiscreteSpace &discrete_space)
    : space_(discrete_space),
      factors_(),
      energy_functions_(),
      constraints_(),
      constraint_functions_(),
      max_factor_arity_(0),
      max_constraint_arity_(0),
      max_factor_size_(0),
      max_constraint_size_(0)
{
}

std::size_t DiscreteGm::add_energy_function(std::unique_ptr<DiscreteEnergyFunctionBase> function)
{
    energy_functions_.push_back(std::move(function));
    return energy_functions_.size() - 1;
}

std::size_t DiscreteGm::add_constraint_function(std::unique_ptr<DiscreteConstraintFunctionBase> function)
{
    constraint_functions_.push_back(std::move(function));
    return constraint_functions_.size() - 1;
}
SolutionValue DiscreteGm::evaluate(const solution_type &solution, bool early_stop_infeasible) const
{
    span<const discrete_label_type> solution_span(solution.data(), solution.size());
    return this->evaluate(solution_span, early_stop_infeasible);
}
SolutionValue DiscreteGm::evaluate(const span<const discrete_label_type> &solution, bool early_stop_infeasible) const
{
    return this->evaluate_if(
        solution, early_stop_infeasible, [](auto) { return true; }, [](auto) { return true; });
}

nlohmann::json DiscreteGm::serialize_json() const
{
    nlohmann::json json_result;
    json_result["model_type"] = "discrete_gm";

    nlohmann::json jgm;

    // json array
    auto jfactors = nlohmann::json::array();
    auto jconstraints = nlohmann::json::array();
    auto jenergy_functions = nlohmann::json::array();
    auto jconstraint_functions = nlohmann::json::array();

    // the space
    jgm["space"] = space_.serialize_json();

    // the energy functions
    for (const auto &energy_function : energy_functions_)
    {
        jenergy_functions.push_back(energy_function->serialize_json());
    }

    // the constraint functions
    for (const auto &constraint_function : constraint_functions_)
    {
        jconstraint_functions.push_back(constraint_function->serialize_json());
    }

    for (const auto &factor : factors_)
    {
        jfactors.push_back({{{"function_index", factor.function_index()}, {"variables", factor.variables()}}});
    }

    for (const auto &constraint : constraints_)
    {
        jconstraints.push_back(
            {{{"function_index", constraint.function_index()}, {"variables", constraint.variables()}}});
    }

    jgm["energy_functions"] = jenergy_functions;
    jgm["constraint_functions"] = jconstraint_functions;
    jgm["factors"] = jfactors;
    jgm["constraints"] = jconstraints;
    json_result["gm"] = jgm;
    return json_result;
}

std::tuple<DiscreteGm, std::unordered_map<std::size_t, std::size_t>, SolutionValue> DiscreteGm::bind(
    span<const uint8_t> mask, span<const discrete_label_type> labels, bool is_include_mask) const
{

    auto [binded_space, gm_to_binded_gm] = space_.subspace(mask, is_include_mask);

    // binded gm
    DiscreteGm binded_gm(binded_space);
    SolutionValue constant_part(0, 0);

    // factors
    for (const auto &factor : factors_)
    {

        // binded variables wrt to the factor
        std::vector<std::size_t> local_binded_vars;
        std::vector<discrete_label_type> local_binded_vars_labels;

        // variable indices wrt to the binded-gm factor
        std::vector<std::size_t> binded_gm_factor_variables;

        for (std::size_t v = 0; v < factor.variables().size(); v++)
        {
            const auto vi = factor.variables()[v];

            if (static_cast<bool>(mask[vi]) != is_include_mask)
            {
                // binded_gm_factor_variables.push_back(gm_to_binded_gm[vi]);
                local_binded_vars.push_back(v);
                local_binded_vars_labels.push_back(labels[vi]);
            }
            else
            {
                binded_gm_factor_variables.push_back(gm_to_binded_gm[vi]);
            }
        }

        NXTGM_CHECK_OP(local_binded_vars.size(), <=, factor.arity(), "");
        NXTGM_CHECK_OP(local_binded_vars.size() + binded_gm_factor_variables.size(), ==, factor.arity(), "");

        if (local_binded_vars.size() > 0 && local_binded_vars.size() < factor.variables().size())
        {
            // some variables are binded
            auto binded_function = factor.function()->bind(
                span<const std::size_t>(local_binded_vars.data(), local_binded_vars.size()),
                span<const discrete_label_type>(local_binded_vars_labels.data(), local_binded_vars_labels.size()));
            auto fid = binded_gm.add_energy_function(std::move(binded_function));
            binded_gm.add_factor(binded_gm_factor_variables, fid);
        }
        else if (local_binded_vars.size() == factor.variables().size())
        {
            // all variables are binded
            constant_part.energy() += factor.function()->energy(local_binded_vars_labels.data());
        }
        else
        {
            // no variables are binded
            auto cloned = factor.function()->clone();
            auto fid = binded_gm.add_energy_function(std::move(cloned));
            binded_gm.add_factor(binded_gm_factor_variables, fid);
        }
    }
    // constraints
    for (const auto &constraint : constraints_)
    {
        // binded variables wrt to the constraint
        std::vector<std::size_t> local_binded_vars;
        std::vector<discrete_label_type> local_binded_vars_labels;

        // variable indices wrt to the binded-gm constraint
        std::vector<std::size_t> binded_gm_constraint_variables;

        for (std::size_t v = 0; v < constraint.variables().size(); v++)
        {
            const auto vi = constraint.variables()[v];
            if (static_cast<bool>(mask[vi]) != is_include_mask)
            {
                local_binded_vars.push_back(v);
                local_binded_vars_labels.push_back(labels[vi]);
            }
            else
            {
                binded_gm_constraint_variables.push_back(gm_to_binded_gm[vi]);
            }
        }

        if (local_binded_vars.size() > 0 && local_binded_vars.size() < constraint.variables().size())
        {
            // some variables are binded
            auto binded_function = constraint.function()->bind(local_binded_vars, local_binded_vars_labels);
            auto cid = binded_gm.add_constraint_function(std::move(binded_function));
            binded_gm.add_constraint(binded_gm_constraint_variables, cid);
        }
        else if (local_binded_vars.size() == constraint.variables().size())
        {
            // all variables are binded
            constant_part.how_violated() += constraint.function()->how_violated(local_binded_vars_labels.data());
        }
        else
        {
            // no variables are binded
            auto cloned = constraint.function()->clone();
            auto cid = binded_gm.add_constraint_function(std::move(cloned));
            binded_gm.add_constraint(binded_gm_constraint_variables, cid);
        }
    }

    return std::tuple<DiscreteGm, std::unordered_map<std::size_t, std::size_t>, SolutionValue>(
        std::move(binded_gm), std::move(gm_to_binded_gm), constant_part);
}

void DiscreteGm::serialize(Serializer &serializer) const
{
    // Serialize the space
    space_.serialize(serializer);

    serializer(energy_functions_.size());
    serializer(constraint_functions_.size());
    serializer(factors_.size());
    serializer(constraints_.size());

    // Serialize the energy functions
    for (const auto &energy_function : energy_functions_)
    {
        energy_function->serialize(serializer);
    }
    // Serialize the constraint functions
    for (const auto &constraint_function : constraint_functions_)
    {
        constraint_function->serialize(serializer);
    }

    // Serialize the factors
    for (const auto &factor : factors_)
    {
        serializer(factor.function_index());
        serializer(factor.variables());
    }
    // Serialize the constraints
    for (const auto &constraint : constraints_)
    {
        serializer(constraint.function_index());
        serializer(constraint.variables());
    }
}

DiscreteGm DiscreteGm::deserialize(Deserializer &deserializer)
{
    // Deserialize the space
    auto space = DiscreteSpace::deserialize(deserializer);

    std::size_t n_energy_functions;
    std::size_t n_constraint_functions;
    std::size_t n_factors;
    std::size_t n_constraints;

    deserializer(n_energy_functions);
    deserializer(n_constraint_functions);
    deserializer(n_factors);
    deserializer(n_constraints);

    // construct gm itself
    DiscreteGm gm(space);

    // energy functions
    for (std::size_t i = 0; i < n_energy_functions; ++i)
    {
        auto energy_function = discrete_energy_function_deserialize(deserializer);
        gm.add_energy_function(std::move(energy_function));
    }

    // constraint functions
    for (std::size_t i = 0; i < n_constraint_functions; ++i)
    {
        auto constraint_function = discrete_constraint_function_deserialize(deserializer);
        gm.add_constraint_function(std::move(constraint_function));
    }

    // factors
    for (std::size_t i = 0; i < n_factors; ++i)
    {
        std::size_t function_index;
        deserializer(function_index);

        std::vector<std::size_t> variables;
        deserializer(variables);
        gm.add_factor(variables, function_index);
    }
    // constraints
    for (std::size_t i = 0; i < n_constraints; ++i)
    {
        std::size_t function_index;
        deserializer(function_index);
        std::vector<std::size_t> variables;
        deserializer(variables);
        gm.add_constraint(variables, function_index);
    }

    return std::move(gm);
}

void DiscreteGm::save_binary(const std::string &path) const
{
    // create some filestream
    std::ofstream os(path, std::ios::binary);
    Serializer serializer(os);

    // serialize
    this->serialize(serializer);
    // close the filestream
    os.close();
}
DiscreteGm DiscreteGm::load_binary(const std::string &path)
{
    std::ifstream is(path, std::ios::binary);
    Deserializer deserializer(is);
    return DiscreteGm::deserialize(deserializer);
}

DiscreteGm DiscreteGm::deserialize_json(const nlohmann::json &json)
{

    if (json["model_type"] != "discrete_gm")
    {
        throw std::runtime_error("json does not contain a discrete_gm");
    }

    const auto jgm = json["model"];
    const auto jspace = jgm["space"];
    const auto jenergy_functions = jgm["energy_functions"];
    const auto jconstraint_functions = jgm["constraint_functions"];
    const auto jfactors = jgm["factors"];
    const auto jconstraints = jgm["constraints"];

    // space
    const auto space = DiscreteSpace::deserialize_json(jgm["space"]);

    // construct gm itself
    DiscreteGm gm(space);

    // energy functions
    for (const auto &jenergy_function : jenergy_functions)
    {
        auto energy_function = discrete_energy_function_deserialize_json(jenergy_function);
        gm.add_energy_function(std::move(energy_function));
    }

    // constraint functions
    for (const auto &jconstraint_function : jconstraint_functions)
    {
        auto constraint_function = discrete_constraint_function_deserialize_json(jconstraint_function);
        gm.add_constraint_function(std::move(constraint_function));
    }

    // factors
    for (const auto &jfactor : jfactors)
    {
        auto function_index = jfactor["function_index"].get<std::size_t>();
        auto variables = jfactor["variables"].get<std::vector<std::size_t>>();
        gm.add_factor(variables, function_index);
    }

    // constraints
    for (const auto &jconstraint : jconstraints)
    {
        auto function_index = jconstraint["function_index"].get<std::size_t>();
        auto variables = jconstraint["variables"].get<std::vector<std::size_t>>();
        gm.add_factor(variables, function_index);
    }

    return std::move(gm);
}
} // namespace nxtgm
