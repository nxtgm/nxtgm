#include <nxtgm/models/gm/discrete_gm.hpp>

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
    : space_(discrete_space), factors_(), energy_functions_(), constraints_(), constraint_functions_(),
      max_factor_arity_(0), max_constraint_arity_(0), max_factor_size_(0), max_constraint_size_(0)
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

    // for the factors we need to figure out the index of the function in the
    // factor wrt the vector (extra scope to delete energy_function_index_map
    // early)
    {
        std::unordered_map<const DiscreteEnergyFunctionBase *, std::size_t> energy_function_index_map;
        for (std::size_t i = 0; i < energy_functions_.size(); i++)
        {
            energy_function_index_map[energy_functions_[i].get()] = i;
        }
        for (const auto &factor : factors_)
        {
            jfactors.push_back({{{"functuin_index", energy_function_index_map[factor.function()]},
                                 {"variables", factor.variables()}}});
        }
    }

    // for the constraints we need to figure out the index of the function in
    // the constrain wrt the vector
    std::unordered_map<const DiscreteConstraintFunctionBase *, std::size_t> constraint_function_index_map;
    for (std::size_t i = 0; i < constraint_functions_.size(); i++)
    {
        constraint_function_index_map[constraint_functions_[i].get()] = i;
    }
    for (const auto &factor : constraints_)
    {
        jconstraints.push_back({{{"functuin_index", constraint_function_index_map[factor.function()]},
                                 {"variables", factor.variables()}}});
    }

    jgm["energy_functions"] = jenergy_functions;
    jgm["constraint_functions"] = jconstraint_functions;
    jgm["factors"] = jfactors;
    jgm["constraints"] = jconstraints;
    json_result["gm"] = jgm;
    return json_result;
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
