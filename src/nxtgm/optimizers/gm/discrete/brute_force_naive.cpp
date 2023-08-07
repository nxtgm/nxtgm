#include <nxtgm/optimizers/gm/discrete/brute_force_naive.hpp>
#include <nxtgm/utils/timer.hpp>

namespace nxtgm
{

BruteForceNaive::BruteForceNaive(const DiscreteGm &gm, const parameters_type &parameters)
    : base_type(gm),
      parameters_(parameters),
      best_solution_(gm.space().size(), 0),
      current_solution_(gm.space().size(), 0),
      best_sol_value_(),
      current_sol_value_()
{

    best_sol_value_ = gm.evaluate(best_solution_, false /* early exit when infeasible*/);
    current_sol_value_ = best_sol_value_;
}

OptimizationStatus BruteForceNaive::optimize(reporter_callback_wrapper_type &reporter_callback,
                                             repair_callback_wrapper_type & /*repair_callback not used*/,
                                             const_discrete_solution_span)
{
    // if the starting point is not empty, use it as the initial solution

    AutoStartedTimer timer;

    OptimizationStatus status = OptimizationStatus::OPTIMAL;

    static const bool early_stop_infeasible = true;

    // indicate the start of the optimization
    reporter_callback.begin();

    // just iterate over all possible solutions:
    // `exitable_for_each_solution` is like
    // `for_each_solution` but it can be exited
    // when the callback returns false.

    bool exit_via_callback = false;
    this->model().space().exitable_for_each_solution(current_solution_, [&](const solution_type &solution) {
        // evaluate the current solution
        this->current_sol_value_ = this->model().evaluate(solution, early_stop_infeasible);

        // if the current solution is better than the best solution
        if (this->current_sol_value_ < this->best_sol_value_)
        {
            this->best_sol_value_ = this->current_sol_value_;
            this->best_solution_ = solution;

            // reporter_callback returns "continue-optimize"
            // and false indicate the optimization should be stopped.
            // we exclude the time spent in the callback from the timer
            // via `paused_call`
            if (reporter_callback && !timer.paused_call([&]() { return reporter_callback.report(); }))
            {
                status = OptimizationStatus::CALLBACK_EXIT;
                return false;
            }
        }

        // check if the time limit is reached
        if (timer.elapsed() > this->parameters_.time_limit)
        {
            status = OptimizationStatus::TIME_LIMIT_REACHED;
            return false;
        }

        // continue the brtue force search
        return true;
    });

    // indicate the end of the optimization
    reporter_callback.end();

    if (!this->best_sol_value_.is_feasible())
    {
        status = OptimizationStatus::INFEASIBLE;
    }

    return status;
}

SolutionValue BruteForceNaive::best_solution_value() const
{
    return this->best_sol_value_;
}
SolutionValue BruteForceNaive::current_solution_value() const
{
    return this->current_sol_value_;
}

const typename BruteForceNaive::solution_type &BruteForceNaive::best_solution() const
{
    return this->best_solution_;
}
const typename BruteForceNaive::solution_type &BruteForceNaive::current_solution() const
{
    return this->current_solution_;
}

} // namespace nxtgm
