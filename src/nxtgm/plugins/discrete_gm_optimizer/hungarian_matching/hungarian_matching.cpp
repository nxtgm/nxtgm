#include <nxtgm/functions/discrete_constraints.hpp>
#include <nxtgm/nxtgm.hpp>
#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>
#include <nxtgm/utils/timer.hpp>

#include "hungarian-algorithm-cpp/Hungarian.h"

namespace nxtgm
{

class HungarianMatching : public DiscreteGmOptimizerBase
{
    class parameters_type
    {
      public:
        inline parameters_type(OptimizerParameters &&parameters)
        {
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
        return "HungarianMatching";
    }
    virtual ~HungarianMatching() = default;

    HungarianMatching(const DiscreteGm &gm, OptimizerParameters &&parameters);

    OptimizationStatus optimize_impl(reporter_callback_wrapper_type &, repair_callback_wrapper_type &,
                                     const_discrete_solution_span) override;

    SolutionValue best_solution_value() const override;
    SolutionValue current_solution_value() const override;

    const solution_type &best_solution() const override;
    const solution_type &current_solution() const override;

  private:
    void check_model() const;

    parameters_type parameters_;
    solution_type best_solution_;
    SolutionValue best_sol_value_;
};

class HungarianMatchingFactory : public DiscreteGmOptimizerFactoryBase
{
  public:
    using factory_base_type = DiscreteGmOptimizerFactoryBase;
    virtual ~HungarianMatchingFactory() = default;
    std::unique_ptr<DiscreteGmOptimizerBase> create(const DiscreteGm &gm, OptimizerParameters &&params) const override
    {
        return std::make_unique<HungarianMatching>(gm, std::move(params));
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
        return "Hungarian  matching algorithm for graphical models "
               "with only unaries and unique label constraints.";
    }
    OptimizerFlags flags() const override
    {
        return OptimizerFlags::OptimalOnTrees;
    }
};
} // namespace nxtgm

XPLUGIN_CREATE_XPLUGIN_FACTORY(nxtgm::HungarianMatchingFactory);

namespace nxtgm
{

HungarianMatching::HungarianMatching(const DiscreteGm &gm, OptimizerParameters &&parameters)
    : base_type(gm, parameters),
      parameters_(std::move(parameters)),
      best_solution_(gm.num_variables(), 0),
      best_sol_value_(gm.evaluate(best_solution_))
{
    // check if the solver is applicable
    this->check_model();

    const auto num_labels = gm.space()[0];
    xt::xtensor<double, 2> cost_matrix = xt::zeros<double>({std::size_t(gm.num_variables()), std::size_t(num_labels)});
    for (const auto &factor : gm.factors())
    {
        auto vi = factor.variables()[0];
        factor.copy_values(&cost_matrix(vi, 0));
    }

    // remove minimum value from cost_matrix
    auto min_value = xt::amin(cost_matrix)();
    cost_matrix -= min_value;

    HungarianAlgorithm hungarian;
    std::vector<int> assignment(gm.num_variables());

    hungarian.Solve(cost_matrix, assignment);

    std::copy(assignment.begin(), assignment.end(), best_solution_.begin());
    best_sol_value_ = gm.evaluate(best_solution_);
}

void HungarianMatching::check_model() const
{
    const auto &gm = this->model();

    if (gm.max_factor_arity() > 2)
    {
        throw std::runtime_error("HungarianMatching only supports factors of arity 2");
    }

    if (gm.num_constraints() != 1 || gm.max_constraint_arity() != gm.num_variables())
    {
        throw std::runtime_error("graphical models needs exactly one global unique label constraint");
    }

    if (!dynamic_cast<const UniqueLables *>(gm.constraints()[0].function()))
    {
        throw std::runtime_error("graphical model has no global unique label constraints");
    }

    if (gm.max_factor_arity() > 1)
    {
        throw std::runtime_error("HungarianMatching only supports unary factors");
    }
    if (!gm.space().is_simple())
    {
        throw std::runtime_error("HungarianMatching only supports simple spaces");
    }
}

OptimizationStatus HungarianMatching::optimize_impl(reporter_callback_wrapper_type &reporter_callback,
                                                    repair_callback_wrapper_type & /*repair_callback not used*/,
                                                    const_discrete_solution_span /*initial_solution not used*/
)
{
    const auto &gm = this->model();
    OptimizationStatus status = OptimizationStatus::OPTIMAL;

    this->best_sol_value_ = gm.evaluate(this->best_solution_);

    return status;
}

SolutionValue HungarianMatching::best_solution_value() const
{
    return this->best_sol_value_;
}
SolutionValue HungarianMatching::current_solution_value() const
{
    return this->best_sol_value_;
}

const typename HungarianMatching::solution_type &HungarianMatching::best_solution() const
{
    return this->best_solution_;
}
const typename HungarianMatching::solution_type &HungarianMatching::current_solution() const
{
    return this->best_solution_;
}
} // namespace nxtgm
