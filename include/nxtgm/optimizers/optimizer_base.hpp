#pragma once

// for inf
#include <limits>

// tuple
#include <tuple>

#include <nxtgm/optimizers/callback_base.hpp>

namespace nxtgm
{

    template<class MODEL_TYPE>
    class OptimizerBase {


    public:
        using self_type = OptimizerBase<MODEL_TYPE>;
        using model_type = MODEL_TYPE;

        using solution_type = typename model_type::solution_type;

        using reporter_callback_base_type = ReporterCallbackBase<self_type>;
        using reporter_callback_wrapper_type = ReporterCallbackWrapper<reporter_callback_base_type>;

        using repair_callback_base_type = RepairCallbackBase<self_type>;
        using repair_callback_wrapper_type = RepairCallbackWrapper<repair_callback_base_type>;
        
        inline OptimizerBase(const model_type & model) : model_(model){
        }

        virtual ~OptimizerBase() = default;
        virtual energy_type lower_bound() const{
            return -std::numeric_limits<energy_type>::infinity();
        }

        virtual SolutionValue best_solution_value() const{
            return this->model()(this->best_solution(), false /* early exit when infeasible*/);
        }

        virtual SolutionValue current_solution_value() const{
            return this->model()(this->best_solution(), false /* early exit when infeasible*/);
        }

        virtual void optimize(
            reporter_callback_base_type * reporter_callback=nullptr,
            repair_callback_base_type * repair_callback=nullptr
        ){
            reporter_callback_wrapper_type reporter_callback_wrapper(reporter_callback) ;
            repair_callback_wrapper_type repair_callback_wrapper(repair_callback);

            this->optimize(reporter_callback_wrapper, repair_callback_wrapper);
        }

        virtual void optimize(
            reporter_callback_wrapper_type & reporter_callback,
            repair_callback_wrapper_type & repair_callback
        ) = 0;

        
        virtual const model_type & model()const{
            return this->model_;
        }
        virtual const solution_type & best_solution() const = 0;
        virtual const solution_type & current_solution() const = 0;
    
    private:
        const model_type & model_;

    };
} 