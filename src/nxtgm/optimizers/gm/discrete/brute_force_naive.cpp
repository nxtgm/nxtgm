#include <nxtgm/optimizers/gm/discrete/brute_force_naive.hpp>

namespace nxtgm
{

    BruteForceNaive::BruteForceNaive(const DiscreteGm & gm, const parameter_type & parameters, const solution_type & initial_solution) 
        : base_type(gm), 
        best_solution_(initial_solution), 
        current_solution_(initial_solution), 
        best_sol_value_(gm(best_solution_, false /* early exit when infeasible*/)),
        current_sol_value_()
    {
        current_sol_value_ = best_sol_value_;
    }

    BruteForceNaive::BruteForceNaive(const DiscreteGm & gm, const parameter_type & parameters) 
        : base_type(gm), 
        best_solution_(gm.space().size(), 0), 
        current_solution_(gm.space().size(), 0), 
        best_sol_value_(gm(best_solution_, false /* early exit when infeasible*/)),
        current_sol_value_()
    {
        current_sol_value_ = best_sol_value_;
    }


    void BruteForceNaive::optimize(
        reporter_callback_wrapper_type & reporter_callback,
        repair_callback_wrapper_type & /*repair_callback not used*/
    ) {
        
        static const bool early_stop_infeasible = true;

        // indicate the start of the optimization
        reporter_callback.begin();

        // just iterate over all possible solutions:
        // `exitable_for_each_solution` is like
        // `for_each_solution` but it can be exited
        // when the callback returns false.
        this->model().space().exitable_for_each_solution(
            current_solution_, 
            [&](const solution_type & solution
        ){
            // evaluate the current solution
            this->current_sol_value_ = this->model()(solution, early_stop_infeasible);

            // if the current solution is better than the best solution
            if(this->current_sol_value_ < this->best_sol_value_)
            {
                this->best_sol_value_ = this->current_sol_value_;
                this->best_solution_ = solution;
            }
            
            // reporter_callback returns "continue-optimize"
            // a falue of false indicate the optimization
            // should be stopped.
            return reporter_callback.report();
        });

        // indicate the end of the optimization
        reporter_callback.end();
    }

    SolutionValue BruteForceNaive::best_solution_value() const {
        return this->best_sol_value_;
    }
    SolutionValue BruteForceNaive::current_solution_value() const {
        return this->current_sol_value_;
    }

    const typename BruteForceNaive::solution_type & BruteForceNaive::best_solution()const {
        return this->best_solution_;
    }
    const typename BruteForceNaive::solution_type & BruteForceNaive::current_solution()const {
        return this->current_solution_;
    }

}