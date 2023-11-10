#include <nxtgm/nxtgm.hpp>
#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>
#include <nxtgm/utils/timer.hpp>

#define ASSERT(x)                                                                                                      \
    if (!(x))                                                                                                          \
    {                                                                                                                  \
        throw std::runtime_error("assertion failed");                                                                  \
    }

namespace nxtgm
{

class DynamicProgramming : public DiscreteGmOptimizerBase
{
    class parameters_type
    {
      public:
        inline parameters_type(OptimizerParameters &&parameters)
        {
            parameters.assign_and_pop_from_any<std::vector<std::size_t>>("roots", roots);
        }
        std::vector<std::size_t> roots;
    };

  public:
    using base_type = DiscreteGmOptimizerBase;
    using solution_type = typename DiscreteGm::solution_type;

    using reporter_callback_wrapper_type = typename base_type::reporter_callback_wrapper_type;
    using repair_callback_wrapper_type = typename base_type::repair_callback_wrapper_type;

    inline static std::string name()
    {
        return "DynamicProgramming";
    }
    virtual ~DynamicProgramming() = default;

    DynamicProgramming(const DiscreteGm &gm, OptimizerParameters &&parameters);

    OptimizationStatus optimize_impl(reporter_callback_wrapper_type &, repair_callback_wrapper_type &,
                                     const_discrete_solution_span) override;

    SolutionValue best_solution_value() const override;
    SolutionValue current_solution_value() const override;

    const solution_type &best_solution() const override;
    const solution_type &current_solution() const override;

  private:
    void compute_labels();

    parameters_type parameters_;
    solution_type best_solution_;
    SolutionValue best_sol_value_;

    DiscreteGmFactorsOfVariables factors_of_variables_;

    std::vector<energy_type> value_buffer_;
    std::vector<discrete_label_type> state_buffer_;
    std::vector<energy_type *> value_buffers_;
    std::vector<discrete_label_type *> state_buffers_;
    std::vector<std::size_t> node_order_;
    std::vector<std::size_t> ordered_nodes_;
};

class DynamicProgrammingFactory : public DiscreteGmOptimizerFactoryBase
{
  public:
    using factory_base_type = DiscreteGmOptimizerFactoryBase;
    virtual ~DynamicProgrammingFactory() = default;
    std::unique_ptr<DiscreteGmOptimizerBase> create(const DiscreteGm &gm, OptimizerParameters &&params) const override
    {
        return std::make_unique<DynamicProgramming>(gm, std::move(params));
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
        return "Dynamic Programming for second order graphical models";
    }
    OptimizerFlags flags() const override
    {
        return OptimizerFlags::OptimalOnTrees;
    }
};
} // namespace nxtgm

XPLUGIN_CREATE_XPLUGIN_FACTORY(nxtgm::DynamicProgrammingFactory);

namespace nxtgm
{

DynamicProgramming::DynamicProgramming(const DiscreteGm &gm, OptimizerParameters &&parameters)
    : base_type(gm, parameters),
      parameters_(std::move(parameters)),
      best_solution_(gm.num_variables()),
      best_sol_value_(gm.evaluate(best_solution_)),
      factors_of_variables_(gm),
      value_buffers_(gm.space().size()),
      state_buffers_(gm.space().size()),
      node_order_(gm.space().size(), std::numeric_limits<std::size_t>::max()),
      ordered_nodes_(gm.space().size(), std::numeric_limits<std::size_t>::max())
{
    ensure_all_handled(name(), parameters);

    if (gm.max_factor_arity() > 2)
    {
        throw UnsupportedModelException("DynamicProgramming only supports factors of arity 2");
    }
    if (!gm.constraints().empty())
    {
        throw UnsupportedModelException("DynamicProgramming does not support constraints");
    }

    // node order
    std::vector<std::size_t> num_children(gm.num_variables(), 0);
    std::vector<std::size_t> node_list;

    std::size_t order_count = 0;
    std::size_t var_count = 0;
    std::size_t root_count = 0;

    constexpr auto mxval = std::numeric_limits<std::size_t>::max();
    while (var_count < gm.num_variables() && order_count < gm.num_variables())
    {
        if (root_count < parameters_.roots.size())
        {
            node_order_[parameters_.roots[root_count]] = order_count++;
            node_list.push_back(parameters_.roots[root_count]);
            ++root_count;
        }
        else if (node_order_[var_count] == std::numeric_limits<std::size_t>::max())
        {
            node_order_[var_count] = order_count++;
            node_list.push_back(var_count);
        }
        ++var_count;
        while (node_list.size() > 0)
        {
            size_t node = node_list.back();
            node_list.pop_back();
            for (auto &&fid : factors_of_variables_[node])
            {
                const auto &factor = gm.factors()[fid];
                const auto &variables = factor.variables();
                if (factor.arity() == 2)
                {
                    if (variables[1] == node && node_order_[variables[0]] == mxval)
                    {
                        node_order_[variables[0]] = order_count++;
                        node_list.push_back(variables[0]);
                        ++num_children[node];
                    }
                    if (variables[0] == node && node_order_[variables[1]] == mxval)
                    {
                        node_order_[variables[1]] = order_count++;
                        node_list.push_back(variables[1]);
                        ++num_children[node];
                    }
                }
            }
        }
    }

    auto buffer_size_values = 0;
    auto buffer_size_states = 0;
    for (std::size_t vi = 0; vi < gm.num_variables(); ++vi)
    {
        buffer_size_values += gm.space()[vi];
        buffer_size_states += gm.space()[vi] * num_children[vi];
    }
    value_buffer_.resize(buffer_size_values);
    state_buffer_.resize(buffer_size_states);

    auto value_ptr = value_buffer_.data();
    auto state_ptr = state_buffer_.data();

    for (std::size_t vi = 0; vi < gm.num_variables(); ++vi)
    {
        value_buffers_[vi] = value_ptr;
        value_ptr += gm.space()[vi];
        state_buffers_[vi] = state_ptr;
        state_ptr += gm.space()[vi] * num_children[vi];
        ordered_nodes_[node_order_[vi]] = vi;
    }
}

OptimizationStatus DynamicProgramming::optimize_impl(reporter_callback_wrapper_type &reporter_callback,
                                                     repair_callback_wrapper_type & /*repair_callback not used*/,
                                                     const_discrete_solution_span /*initial_solution not used*/
)
{
    const auto &gm = this->model();
    OptimizationStatus status = OptimizationStatus::OPTIMAL;

    for (std::size_t i = 1; i <= gm.num_variables(); ++i)
    {

        const auto node = ordered_nodes_[gm.num_variables() - i];

        std::fill(value_buffers_[node], value_buffers_[node] + gm.num_labels(node), energy_type(0));

        // accumulate messages
        std::size_t children_counter = 0;
        for (auto fid : factors_of_variables_[node])
        {
            auto &&factor = gm.factors()[fid];

            // unary
            if (factor.arity() == 1)
            {
                factor.add_values(value_buffers_[node]);
            }
            // pairwise
            if (factor.arity() == 2)
            {

                auto &&vars = factor.variables();
                if (vars[0] == node && node_order_[vars[1]] > node_order_[node])
                {
                    const auto node2 = vars[1];
                    discrete_label_type s;
                    energy_type v;
                    for (discrete_label_type l0 = 0; l0 < gm.num_labels(node); ++l0)
                    {
                        v = std::numeric_limits<energy_type>::infinity();
                        for (discrete_label_type l1 = 0; l1 < gm.num_labels(node2); ++l1)
                        {
                            const auto factor_value = factor({l0, l1});
                            const auto v2 = factor_value + value_buffers_[node2][l1];
                            if (v2 < v)
                            {
                                v = v2;
                                s = l1;
                            }
                        }
                        state_buffers_[node][children_counter * gm.num_labels(node) + l0] = s;
                        value_buffers_[node][l0] += v;
                    }
                    ++children_counter;
                }
                if (vars[1] == node && node_order_[vars[0]] > node_order_[node])
                {
                    const auto node2 = vars[0];
                    for (discrete_label_type l1 = 0; l1 < gm.num_labels(node); ++l1)
                    {
                        discrete_label_type s;
                        auto v = std::numeric_limits<energy_type>::infinity();

                        for (discrete_label_type l0 = 0; l0 < gm.num_labels(node2); ++l0)
                        {

                            const auto v2 = factor({l0, l1}) + value_buffers_[node2][l0];
                            if (v2 < v)
                            {
                                v = v2;
                                s = l0;
                            }
                        }
                        state_buffers_[node][children_counter * gm.num_labels(node) + l1] = s;
                        value_buffers_[node][l1] += v;
                    }
                    ++children_counter;
                }
            }
        }
    }

    this->compute_labels();

    this->best_sol_value_ = gm.evaluate(this->best_solution_);

    return status;
}

SolutionValue DynamicProgramming::best_solution_value() const
{
    return this->best_sol_value_;
}
SolutionValue DynamicProgramming::current_solution_value() const
{
    return this->best_sol_value_;
}

const typename DynamicProgramming::solution_type &DynamicProgramming::best_solution() const
{
    return this->best_solution_;
}
const typename DynamicProgramming::solution_type &DynamicProgramming::current_solution() const
{
    return this->best_solution_;
}

void DynamicProgramming::compute_labels()
{

    const auto &gm = this->model();

    std::vector<std::size_t> node_list;
    std::fill(best_solution_.begin(), best_solution_.end(), std::numeric_limits<discrete_label_type>::max());
    std::size_t var = 0;
    while (var < gm.num_variables())
    {
        if (best_solution_[var] == std::numeric_limits<discrete_label_type>::max())
        {
            energy_type v = std::numeric_limits<energy_type>::infinity();
            for (std::size_t i = 0; i < gm.num_labels(var); ++i)
            {
                if (value_buffers_[var][i] < v)
                {
                    v = value_buffers_[var][i];
                    best_solution_[var] = i;
                }
            }
            node_list.push_back(var);
        }
        ++var;
        while (node_list.size() > 0)
        {
            std::size_t node = node_list.back();
            std::size_t children_counter = 0;
            node_list.pop_back();

            for (auto &&fid : factors_of_variables_[node])
            {
                auto &&factor = gm.factors()[fid];
                auto &&vars = factor.variables();

                if (factor.arity() == 2)
                {
                    if (vars[1] == node && node_order_[vars[0]] > node_order_[node])
                    {
                        best_solution_[vars[0]] =
                            state_buffers_[node][children_counter * gm.num_labels(node) + best_solution_[node]];
                        node_list.push_back(vars[0]);
                        ++children_counter;
                    }
                    if (vars[0] == node && node_order_[vars[1]] > node_order_[node])
                    {
                        best_solution_[vars[1]] =
                            state_buffers_[node][children_counter * gm.num_labels(node) + best_solution_[node]];
                        node_list.push_back(vars[1]);
                        ++children_counter;
                    }
                }
            }
        }
    }
}

} // namespace nxtgm
