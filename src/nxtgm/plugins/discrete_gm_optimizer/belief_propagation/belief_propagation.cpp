#include <limits>
#include <nxtgm/nxtgm.hpp>
#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>
#include <nxtgm/utils/timer.hpp>
#include <random>

namespace nxtgm
{

class BeliefPropagation : public DiscreteGmOptimizerBase
{
    class parameters_type
    {
      public:
        using belief_callack_type = std::function<void(const energy_type *beliefs)>;

        inline parameters_type(OptimizerParameters &parameters)
        {
            parameters.assign_and_pop("max_iterations", max_iterations);
            parameters.assign_and_pop("convergence_tolerance", convergence_tolerance);
            parameters.assign_and_pop("damping", damping);
            parameters.assign_and_pop("normalize_messages", normalize_messages);
            parameters.assign_and_pop("report_beliefs", report_beliefs);
            parameters.assign_and_pop("constraint_scaling", constraint_scaling);
            parameters.assign_and_pop("compute_message_parameters", compute_message_parameters);
            parameters.assign_and_pop("random_initialization", random_initialization);
            parameters.assign_and_pop("seed", seed);
        }

        std::size_t max_iterations = 1000;
        energy_type convergence_tolerance = 1e-5;
        energy_type damping = 0.0;
        energy_type constraint_scaling = default_constraint_scaling;
        bool normalize_messages = true;
        bool report_beliefs = false;
        bool random_initialization = false;
        int seed = 0;

        // this is passed to the constraint functions
        // since there are complicated constraint functions
        // which need parametrization (ie high arity UniqueLabelConstraints)
        OptimizerParameters compute_message_parameters;
    };

  public:
    using base_type = DiscreteGmOptimizerBase;
    using solution_type = typename DiscreteGm::solution_type;

    using reporter_callback_wrapper_type = typename base_type::reporter_callback_wrapper_type;
    using repair_callback_wrapper_type = typename base_type::repair_callback_wrapper_type;

    inline static std::string name()
    {
        return "BeliefPropagation";
    }
    virtual ~BeliefPropagation() = default;

    BeliefPropagation(const DiscreteGm &gm, OptimizerParameters &&json_parameters);

    OptimizationStatus optimize_impl(reporter_callback_wrapper_type &, repair_callback_wrapper_type &,
                                     const_discrete_solution_span starting_point) override;

    SolutionValue best_solution_value() const override;
    SolutionValue current_solution_value() const override;

    const solution_type &best_solution() const override;
    const solution_type &current_solution() const override;

    void normalize_message(energy_type *begin, energy_type *end) const;

  private:
    // variable_to_factor / variable_to_constraint messages
    void compute_from_variable_messages();

    // factor_to_variable / constraint_to_variable messages
    void compute_to_variable_messages();
    void compute_beliefs();
    void compute_solution();

    energy_type compute_convergence_delta();
    void damp_messages();

    parameters_type parameters_;
    std::size_t iteration_;

    // since we do damping, we need to store old and current messages
    std::vector<energy_type> message_storage_[2];

    // old and new beliefs
    std::vector<energy_type> belief_storage_;

    // offsets for the messages
    std::vector<std::size_t> factor_to_variable_message_offsets_;
    std::vector<std::size_t> variable_to_factor_message_offsets_;
    std::vector<std::size_t> constraint_to_variable_message_offsets_;
    std::vector<std::size_t> variable_to_constraint_message_offsets_;
    std::vector<std::size_t> belief_offsets_;

    std::vector<energy_type *> local_to_variable_messages_;
    std::vector<energy_type *> local_from_variable_messages_;

    std::vector<discrete_label_type> max_arity_label_buffer;

    SolutionValue best_solution_value_;
    SolutionValue current_solution_value_;
    solution_type best_solution_;
    solution_type current_solution_;
};

class BeliefPropagationFactory : public DiscreteGmOptimizerFactoryBase
{
  public:
    using factory_base_type = DiscreteGmOptimizerFactoryBase;
    virtual ~BeliefPropagationFactory() = default;
    expected<std::unique_ptr<DiscreteGmOptimizerBase>> create(const DiscreteGm &gm,
                                                              OptimizerParameters &&params) const override
    {
        return std::make_unique<BeliefPropagation>(gm, std::move(params));
    }
    int priority() const override
    {
        return plugin_priority(PluginPriority::MEDIUM);
    }
    std::string license() const override
    {
        return "MIT";
    }
    std::string description() const override
    {
        return "BeliefPropagation with parallel message passing update rules";
    }
};

} // namespace nxtgm

XPLUGIN_CREATE_XPLUGIN_FACTORY(nxtgm::BeliefPropagationFactory);

namespace nxtgm
{

BeliefPropagation::BeliefPropagation(const DiscreteGm &gm, OptimizerParameters &&parameters)
    : base_type(gm, parameters),
      parameters_(parameters),
      iteration_(0),
      message_storage_(),
      belief_storage_(),
      factor_to_variable_message_offsets_(gm.num_factors()),
      variable_to_factor_message_offsets_(gm.num_factors()),
      constraint_to_variable_message_offsets_(gm.num_constraints()),
      variable_to_constraint_message_offsets_(gm.num_constraints()),
      belief_offsets_(gm.num_variables()),
      local_to_variable_messages_(gm.max_arity()),
      local_from_variable_messages_(gm.max_arity()),
      max_arity_label_buffer(gm.max_arity()),
      best_solution_value_(),
      current_solution_value_(),
      best_solution_(gm.num_variables(), 0),
      current_solution_(gm.num_variables(), 0)
{
    ensure_all_handled(name(), parameters);

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

    // messages for factors
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

    // messages for constraints
    for (std::size_t ci = 0; ci < gm.num_constraints(); ++ci)
    {
        std::size_t size_for_constraint = 0;
        for (auto &variable : gm.constraint(ci).variables())
        {
            size_for_constraint += gm.num_labels(variable);
        }
        variable_to_constraint_message_offsets_[ci] = message_storage_size;
        constraint_to_variable_message_offsets_[ci] = message_storage_size + size_for_constraint;
        message_storage_size += 2 * size_for_constraint;
    }

    message_storage_[0].resize(message_storage_size, 0);
    message_storage_[1].resize(message_storage_size, 0);

    if (parameters_.random_initialization)
    {
        std::mt19937 gen(parameters_.seed);
        std::uniform_real_distribution<> dis(0.0, 1.0);
        for (auto &m : message_storage_[0])
        {
            m = dis(gen);
        }
    }
}

OptimizationStatus BeliefPropagation::optimize_impl(reporter_callback_wrapper_type &reporter_callback,
                                                    repair_callback_wrapper_type & /*repair_callback not used*/,
                                                    const_discrete_solution_span starting_point)
{
    // shortcut to the model
    const auto &gm = this->model();

    ReportData report_data;
    report_data.double_data["beliefs"] = span<const energy_type>(belief_storage_.data(), belief_storage_.size());

    if (starting_point.size() > 0)
    {
        std::copy(starting_point.begin(), starting_point.end(), best_solution_.begin());
        best_solution_value_ = gm.evaluate(best_solution_, false /* early exit when infeasible*/);
    }

    for (std::size_t i = 0; i < parameters_.max_iterations; ++i)
    {
        this->iteration_ = i;
        this->compute_from_variable_messages();
        this->compute_to_variable_messages();
        this->damp_messages();
        this->compute_beliefs();
        this->compute_solution();

        auto delta = this->compute_convergence_delta();
        if (delta < parameters_.convergence_tolerance)
        {
            return OptimizationStatus::CONVERGED;
        }

        // copy the messages
        std::copy(message_storage_[0].begin(), message_storage_[0].end(), message_storage_[1].begin());

        if (!this->report(reporter_callback, report_data))
        {
            return OptimizationStatus::CALLBACK_EXIT;
        }
        // check if the time limit is reached
        if (this->time_limit_reached())
        {
            return OptimizationStatus::TIME_LIMIT_REACHED;
        }
    }

    return OptimizationStatus::ITERATION_LIMIT_REACHED;
}

void BeliefPropagation::compute_beliefs()
{
    // shortcut to the model
    const auto &gm = this->model();

    // reset the beliefs
    std::fill(belief_storage_.begin(), belief_storage_.end(), 0);

    // compute beliefs
    // - from factors
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
    // - from constraints
    for (std::size_t ci = 0; ci < gm.num_constraints(); ++ci)
    {
        auto constraint_to_var_ptr = message_storage_[0].data() + constraint_to_variable_message_offsets_[ci];
        for (auto &&variable : gm.constraint(ci).variables())
        {

            auto belief_ptr = belief_storage_.data() + belief_offsets_[variable];

            const auto num_labels = gm.num_labels(variable);

            // the explicit for loop is more readable then the std::transform
            for (discrete_label_type label = 0; label < num_labels; ++label)
            {
                belief_ptr[label] += constraint_to_var_ptr[label];
            }

            constraint_to_var_ptr += num_labels;
        }
    }
}

void BeliefPropagation::normalize_message(energy_type *begin, energy_type *end) const
{
    // find min
    const auto min = *std::min_element(begin, end);

    // subtract min
    for (std::size_t label = 0; label < end - begin; ++label)
    {
        begin[label] -= min;
    }
}

// variable_to_factor / variable_to_factor messages
void BeliefPropagation::compute_from_variable_messages()
{
    // a bit stupid that code for constraints and factors is *almost* the same

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
    for (std::size_t ci = 0; ci < gm.num_constraints(); ++ci)
    {
        // messages associated with unary constraints
        // do not change  after the first iteration
        const auto &constraint = gm.constraint(ci);
        if (iteration_ > 0 && constraint.arity() == 1)
        {
            continue;
        }

        auto var_to_constraint_ptr = message_storage_[0].data() + variable_to_constraint_message_offsets_[ci];
        auto constraint_to_var_ptr = message_storage_[0].data() + constraint_to_variable_message_offsets_[ci];

        for (auto variable : constraint.variables())
        {
            auto belief_ptr = belief_storage_.data() + belief_offsets_[variable];

            const auto num_labels = gm.num_labels(variable);

            for (discrete_label_type label = 0; label < num_labels; ++label)
            {
                var_to_constraint_ptr[label] = belief_ptr[label] - constraint_to_var_ptr[label];
            }

            var_to_constraint_ptr += num_labels;
            constraint_to_var_ptr += num_labels;
        }
    }
}

// factor_to_variable / constraint_to_variable messages
void BeliefPropagation::compute_to_variable_messages()
{
    const auto &gm = this->model();
    for (std::size_t fi = 0; fi < gm.num_factors(); ++fi)
    {
        const auto &factor = gm.factor(fi);

        // messages associated with unary factors  do not change  after the first iteration
        if (iteration_ > 0 && factor.arity() == 1)
        {
            continue;
        }

        auto var_to_fac_ptr = message_storage_[0].data() + variable_to_factor_message_offsets_[fi];
        auto fac_to_var_ptr = message_storage_[0].data() + factor_to_variable_message_offsets_[fi];

        for (std::size_t ai = 0; ai < factor.arity(); ++ai)
        {
            const auto num_lables = gm.num_labels(factor.variables()[ai]);
            local_to_variable_messages_[ai] = fac_to_var_ptr;
            local_from_variable_messages_[ai] = var_to_fac_ptr;
            fac_to_var_ptr += num_lables;
            var_to_fac_ptr += num_lables;
        }

        // compute the messages
        factor.function()->compute_to_variable_messages(
            static_cast<const energy_type *const *>(local_from_variable_messages_.data()),
            local_to_variable_messages_.data());

        // normalize the messages
        if (parameters_.normalize_messages)
        {
            for (std::size_t ai = 0; ai < factor.arity(); ++ai)
            {
                this->normalize_message(local_to_variable_messages_[ai],
                                        local_to_variable_messages_[ai] + gm.num_labels(factor.variables()[ai]));
            }
        }
    }

    for (std::size_t ci = 0; ci < gm.num_constraints(); ++ci)
    {
        const auto &constraint = gm.constraint(ci);

        // messages associated with unary constraints do not change  after the first iteration
        if (iteration_ > 0 && constraint.arity() == 1)
        {
            continue;
        }

        auto var_to_ptr = message_storage_[0].data() + variable_to_constraint_message_offsets_[ci];
        auto constraint_to_var_ptr = message_storage_[0].data() + constraint_to_variable_message_offsets_[ci];

        for (std::size_t ai = 0; ai < constraint.arity(); ++ai)
        {

            const auto num_lables = gm.num_labels(constraint.variables()[ai]);
            local_to_variable_messages_[ai] = constraint_to_var_ptr;
            local_from_variable_messages_[ai] = var_to_ptr;
            constraint_to_var_ptr += num_lables;
            var_to_ptr += num_lables;
        }

        // compute the messages
        constraint.function()->compute_to_variable_messages(
            static_cast<const energy_type *const *>(local_from_variable_messages_.data()),
            local_to_variable_messages_.data(), parameters_.constraint_scaling, parameters_.compute_message_parameters);

        // normalize the messages
        if (parameters_.normalize_messages)
        {
            for (std::size_t ai = 0; ai < constraint.arity(); ++ai)
            {
                this->normalize_message(local_to_variable_messages_[ai],
                                        local_to_variable_messages_[ai] + gm.num_labels(constraint.variables()[ai]));
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
