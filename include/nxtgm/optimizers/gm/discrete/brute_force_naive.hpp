#pragma once

#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>

namespace nxtgm
{

    class BruteForceNaive : public DiscreteGmOptimizerBase{
    public:
        
        class parameter_type{
        };

        using base_type = DiscreteGmOptimizerBase;
        using solution_type = typename DiscreteGm::solution_type;

        using reporter_callback_wrapper_type = typename base_type::reporter_callback_wrapper_type;
        using repair_callback_wrapper_type = typename base_type::repair_callback_wrapper_type;
        
        using base_type::optimize;

        inline static std::string name()
        {
            return "BruteForceNaive";
        }
        virtual ~BruteForceNaive() = default;


        BruteForceNaive(const DiscreteGm & gm, const parameter_type & parameters, const solution_type & initial_solution);
        BruteForceNaive(const DiscreteGm & gm, const parameter_type & parameters);

        virtual void optimize(reporter_callback_wrapper_type & ,repair_callback_wrapper_type &) override;

        SolutionValue best_solution_value() const override;
        SolutionValue current_solution_value() const override;

        const solution_type & best_solution()const override;
        const solution_type & current_solution()const override;
        
    private:
        solution_type best_solution_;
        solution_type current_solution_;
        SolutionValue best_sol_value_;
        SolutionValue current_sol_value_;
    };
}
