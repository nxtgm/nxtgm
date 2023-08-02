#include <iostream>
#include <limits>
#include <nxtgm/nxtgm.hpp>
#include <nxtgm/optimizers/gm/discrete/belief_propagation.hpp>
#include <nxtgm/utils/timer.hpp>

namespace nxtgm
{

BeliefPropagation::BeliefPropagation(const DiscreteGm &gm, const parameters_type &parameters)
    : base_type(gm), parameters_(parameters), iteration_(0), message_storage_(), belief_storage_(),
      factor_to_variable_message_offsets_(gm.num_factors()), variable_to_factor_message_offsets_(gm.num_factors()),
      belief_offsets_(gm.num_variables()), local_factor_to_variable_messages_(gm.max_arity()),
      local_variable_to_factor_messages_(gm.max_arity()), max_arity_label_buffer(gm.max_arity()),
      best_solution_value_(), current_solution_value_(), best_solution_(gm.num_variables(), 0),
      current_solution_(gm.num_variables(), 0)
{
    current_solution_value_ = gm.evaluate(current_solution_, false /* early exit when infeasible*/);
    best_solution_value_ = current_solution_value_;

    // belief
    std::size_t belief_storage_size = 0;
    for (std::size_t vi = 0; vi < gm.num_variables(); ++vi)
    {
        belief_offsets_[vi] = belief_storage_size;
        belief_storage_size += gm.num_labels(vi);
    }
    belief_storage_.resize(belief_storage_size, 0);

    // messages
    std::size_t message_storage_size = 0;
    for (std::size_t fi = 0; fi < gm.num_factors(); ++fi)
    {

        std::size_t size_for_factor = 0;
        for (auto &variable : gm.factor(fi).variables())
        {
            size_for_factor += gm.num_labels(variable);
        }
        variable_to_factor_message_offsets_[fi] = message_storage_size;
        factor_to_variable_message_offsets_[fi] = message_storage_size + size_for_factor;
        message_storage_size += 2 * size_for_factor;
    }
    message_storage_[0].resize(message_storage_size, 0);
    message_storage_[1].resize(message_storage_size, 0);
}

OptimizationStatus BeliefPropagation::optimize(reporter_callback_wrapper_type &reporter_callback,
                                               repair_callback_wrapper_type & /*repair_callback not used*/,
                                               const_discrete_solution_span)
{
    // std::cout<<"BeliefPropagation::optimize"<<std::endl;
    //  indicate the start of the optimization
    reporter_callback.begin();

    // start the timer
    AutoStartedTimer timer;

    // shortcut to the model
    const auto &gm = this->model();

    // report the current solution to callack
    if (reporter_callback && !timer.paused_call([&]() { return reporter_callback.report(); }))
    {
        return OptimizationStatus::CALLBACK_EXIT;
    }

    for (iteration_ = 0; iteration_ < parameters_.max_iterations; ++iteration_)
    {
        this->compute_variable_to_factor_messages();
        this->compute_factor_to_variable_messages();
        this->damp_messages();
        this->compute_beliefs();
        this->compute_solution();

        auto delta = this->compute_convergence_delta();
        if (delta < parameters_.convergence_tolerance)
        {
            reporter_callback.end();
            return OptimizationStatus::CONVERGED;
        }

        // copy the messages
        std::copy(message_storage_[0].begin(), message_storage_[0].end(), message_storage_[1].begin());

        // check if the time limit is reached
        if (timer.elapsed() > this->parameters_.time_limit)
        {
            reporter_callback.end();
            return OptimizationStatus::TIME_LIMIT_REACHED;
        }
    }

    // indicate the end of the optimization
    reporter_callback.end();
    return OptimizationStatus::ITERATION_LIMIT_REACHED;
}

void BeliefPropagation::compute_beliefs()
{
    // shortcut to the model
    const auto &gm = this->model();

    // reset the beliefs
    std::fill(belief_storage_.begin(), belief_storage_.end(), 0);

    // compute beliefs
    for (std::size_t fi = 0; fi < gm.num_factors(); ++fi)
    {

        auto fac_to_var_ptr = message_storage_[0].data() + factor_to_variable_message_offsets_[fi];
        for (auto &&variable : gm.factor(fi).variables())
        {

            auto belief_ptr = belief_storage_.data() + belief_offsets_[variable];

            const auto num_labels = gm.num_labels(variable);

            // the explicit for loop is more readable then the std::transform
            for (discrete_label_type label = 0; label < num_labels; ++label)
            {
                belief_ptr[label] += fac_to_var_ptr[label];
            }

            fac_to_var_ptr += num_labels;
        }
    }
}

void BeliefPropagation::compute_variable_to_factor_messages()
{
    const auto &gm = this->model();
    for (std::size_t fi = 0; fi < gm.num_factors(); ++fi)
    {
        // messages associated with unary factors
        // do not change  after the first iteration
        const auto &factor = gm.factor(fi);
        if (iteration_ > 0 && factor.arity() == 1)
        {
            continue;
        }

        auto var_to_fac_ptr = message_storage_[0].data() + variable_to_factor_message_offsets_[fi];
        auto fac_to_var_ptr = message_storage_[0].data() + factor_to_variable_message_offsets_[fi];

        for (auto variable : factor.variables())
        {
            auto belief_ptr = belief_storage_.data() + belief_offsets_[variable];

            const auto num_labels = gm.num_labels(variable);

            for (discrete_label_type label = 0; label < num_labels; ++label)
            {
                var_to_fac_ptr[label] = belief_ptr[label] - fac_to_var_ptr[label];
            }

            var_to_fac_ptr += num_labels;
            fac_to_var_ptr += num_labels;
        }
    }
}

void BeliefPropagation::compute_factor_to_variable_messages()
{
    const auto &gm = this->model();
    for (std::size_t fi = 0; fi < gm.num_factors(); ++fi)
    {

        const auto &factor = gm.factor(fi);

        // messages associated with unary factors
        // do not change  after the first iteration
        if (iteration_ > 0 && factor.arity() == 1)
        {
            continue;
        }

        auto var_to_fac_ptr = message_storage_[0].data() + variable_to_factor_message_offsets_[fi];
        auto fac_to_var_ptr = message_storage_[0].data() + factor_to_variable_message_offsets_[fi];

        for (std::size_t ai = 0; ai < factor.arity(); ++ai)
        {

            const auto num_lables = gm.num_labels(factor.variables()[ai]);
            local_factor_to_variable_messages_[ai] = fac_to_var_ptr;
            local_variable_to_factor_messages_[ai] = var_to_fac_ptr;

            fac_to_var_ptr += num_lables;
            var_to_fac_ptr += num_lables;
        }

        const auto &local_variable_to_factor_messages = local_variable_to_factor_messages_;

        // compute the messages
        factor.function()->compute_factor_to_variable_messages(
            static_cast<const energy_type *const *>(local_variable_to_factor_messages.data()),
            local_factor_to_variable_messages_.data());

        // normalize the messages
        if (parameters_.normalize_messages)
        {
            for (std::size_t ai = 0; ai < factor.arity(); ++ai)
            {
                // find min
                const auto min =
                    *std::min_element(local_factor_to_variable_messages_[ai],
                                      local_factor_to_variable_messages_[ai] + gm.num_labels(factor.variables()[ai]));
                // subtract min
                for (std::size_t label = 0; label < gm.num_labels(factor.variables()[ai]); ++label)
                {
                    local_factor_to_variable_messages_[ai][label] -= min;
                }
            }
        }
    }
}

void BeliefPropagation::compute_solution()
{

    // compute the solution
    const auto &gm = this->model();
    for (std::size_t vi = 0; vi < gm.num_variables(); ++vi)
    {
        const auto num_labels = gm.num_labels(vi);
        auto belief_ptr = belief_storage_.data() + belief_offsets_[vi];
        const auto best_label = std::distance(belief_ptr, std::min_element(belief_ptr, belief_ptr + num_labels));
        current_solution_[vi] = best_label;
    }

    // eval the solution
    current_solution_value_ = gm.evaluate(current_solution_, false /* early exit when infeasible*/);
    if (current_solution_value_ < best_solution_value_)
    {
        best_solution_value_ = current_solution_value_;
        best_solution_ = current_solution_;
    }
}

energy_type BeliefPropagation::compute_convergence_delta()
{
    if (iteration_ == 0)
    {
        return std::numeric_limits<energy_type>::infinity();
    }
    else
    {
        // accumulate element wise squared distance
        // between old and new messages
        // ie message_storage_[0] and message_storage_[1]
        energy_type acc = 0;
        for (std::size_t i = 0; i < message_storage_[0].size(); ++i)
        {
            acc += std::pow(message_storage_[0][i] - message_storage_[1][i], 2);
        }

        return std::sqrt(acc);
    }
}

void BeliefPropagation::damp_messages()
{
    if (iteration_ == 0 || parameters_.damping < std::numeric_limits<energy_type>::epsilon())
    {
        return;
    }
    for (std::size_t i = 0; i < message_storage_[0].size(); ++i)
    {
        message_storage_[0][i] =
            (1 - parameters_.damping) * message_storage_[0][i] + parameters_.damping * message_storage_[1][i];
    }
}

SolutionValue BeliefPropagation::best_solution_value() const
{
    return best_solution_value_;
}
SolutionValue BeliefPropagation::current_solution_value() const
{
    return current_solution_value_;
}

const typename BeliefPropagation::solution_type &BeliefPropagation::best_solution() const
{
    return best_solution_;
}
const typename BeliefPropagation::solution_type &BeliefPropagation::current_solution() const
{
    return current_solution_;
}

} // namespace nxtgm
