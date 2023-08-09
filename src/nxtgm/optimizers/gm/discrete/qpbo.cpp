#include <iostream>
#include <limits>
#include <nxtgm/nxtgm.hpp>
#include <nxtgm/optimizers/gm/discrete/qpbo.hpp>
#include <nxtgm/plugins/plugin_registry.hpp>
#include <nxtgm/utils/timer.hpp>

namespace nxtgm
{

Qpbo::Qpbo(const DiscreteGm &gm, const parameters_type &parameters)
    : base_type(gm),
      parameters_(parameters),
      best_solution_value_(),
      best_solution_(gm.num_variables(), 0),
      qpbo_(nullptr),
      qpbo_labels_(gm.num_variables(), -1)
{

    best_solution_value_ = gm.evaluate(best_solution_, false /* early exit when infeasible*/);

    // check space
    if (gm.space().max_num_labels() != 2)
    {
        throw std::runtime_error("QPBO only supports binary variables");
    }
    // check for constraints
    if (gm.num_constraints() > 0)
    {
        throw std::runtime_error("QPBO does not support constraints");
    }
    // check arity
    if (gm.max_arity() > 2)
    {
        throw std::runtime_error("QPBO only supports pairwise factors");
    }

    // build the qpbo model count the number of edges
    size_t num_edges = 0;
    for (const auto &factor : gm.factors())
    {
        if (const auto arity = factor.arity(); arity == 2)
            num_edges += 1;
    }
    auto factory = get_plugin_registry<QpboFactoryBase>().get_factory(parameters_.qpbo_plugin_name);
    qpbo_ = factory->create(gm.num_variables(), num_edges);

    double energies[4] = {0, 0, 0, 0};
    for (size_t i = 0; i < gm.num_factors(); ++i)
    {
        const auto &factor = gm.factor(i);
        if (const auto arity = factor.arity(); arity == 1)
        {
            factor.copy_energies(energies);
            qpbo_->add_unary_term(factor.variables()[0], energies);
        }
        else if (arity == 2)
        {
            factor.copy_energies(energies);
            qpbo_->add_pairwise_term(factor.variables()[0], factor.variables()[1], energies);
        }
    }
}

OptimizationStatus Qpbo::optimize(reporter_callback_wrapper_type &reporter_callback,
                                  repair_callback_wrapper_type & /*repair_callback not used*/,
                                  const_discrete_solution_span)
{

    reporter_callback.begin();

    // start the timer
    AutoStartedTimer timer;

    // shortcut to the model
    const auto &gm = this->model();

    qpbo_->solve(qpbo_labels_.data());

    std::size_t num_unlabeled = 0;
    for (std::size_t vi = 0; vi < gm.num_variables(); ++vi)
    {
        const auto qpbo_label = qpbo_labels_[vi];
        if (qpbo_label < 0)
        {
            ++num_unlabeled;
            best_solution_[vi] = 0;
        }
        else
        {
            best_solution_[vi] = qpbo_label;
        }
    }

    best_solution_value_ = gm.evaluate(best_solution_, false /* early exit when infeasible*/);

    // indicate the end of the optimization
    reporter_callback.end();

    return num_unlabeled == 0 ? OptimizationStatus::OPTIMAL : OptimizationStatus::CONVERGED;
}

SolutionValue Qpbo::best_solution_value() const
{
    return best_solution_value_;
}
SolutionValue Qpbo::current_solution_value() const
{
    return best_solution_value_;
}

const typename Qpbo::solution_type &Qpbo::best_solution() const
{
    return best_solution_;
}
const typename Qpbo::solution_type &Qpbo::current_solution() const
{
    return best_solution_;
}

bool Qpbo::is_partial_optimal(std::size_t variable_index) const
{
    return qpbo_labels_[variable_index] >= 0;
}

} // namespace nxtgm
