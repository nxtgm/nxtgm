
#include <nxtgm/optimizers/gm/discrete/ilp_highs.hpp>
#include <nxtgm/utils/n_nested_loops.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>

#include <nxtgm/utils/lp.hpp>

namespace nxtgm{

    IlpHighs::IlpHighs(const DiscreteGm & gm, const parameter_type & parameters, const solution_type & initial_solution) 
    :   base_type(gm), 
        best_solution_(), 
        current_solution_(), 
        best_sol_value_(),
        current_sol_value_(),
        indicator_variable_mapping_(gm.space())
    {
        if(initial_solution.empty()){
            best_solution_ = solution_type(gm.space().size());
        }else{
            best_solution_ = initial_solution;
        }
        best_sol_value_ = this->model()(best_solution_, false);
        current_solution_ = best_solution_;
        current_sol_value_ = best_sol_value_;

        this->setup_lp();
    }

    void IlpHighs::setup_lp()
    {
        
        // shortcuts
        const auto   & model = this->model();
        const auto & space = model.space();

        // buffers
        IlpData ilp_data;

        // add inter variables for all the indicator variables
        // (objective will be added later)
        ilp_data.add_variables(indicator_variable_mapping_.num_indicator_variables() , 0.0, 1.0, 0.0, true);

        // sum to one constraints
        for(std::size_t vi = 0; vi <space.size(); ++vi){
            ilp_data.begin_constraint(1.0, 1.0);
            for(discrete_label_type l=0; l < space[vi]; ++l){
                ilp_data.add_constraint_coefficient(indicator_variable_mapping_[vi] + l, 1.0);
            }
        }

        // add all the factors to the ilp
        IlpFactorBuilderBuffer ilp_factor_builder_buffer;
        std::vector<std::size_t> indicator_variables_mapping_buffer(model.max_arity());
       
        for(auto && factor : model.factors()){
            factor.map_from_model(indicator_variable_mapping_, indicator_variables_mapping_buffer);
            span<std::size_t> factor_indicator_variables_mapping(indicator_variables_mapping_buffer.data(), factor.arity());
            factor.function()->add_to_lp(ilp_data, factor_indicator_variables_mapping, ilp_factor_builder_buffer);
        };


        // add constraints to the ilp
        IlpConstraintBuilderBuffer ilp_constraint_builder_buffer;
        for(auto  && constraint : model.constraints())
        {
            constraint.map_from_model(indicator_variable_mapping_, indicator_variables_mapping_buffer);   
            span<std::size_t> constraint_indicator_variables_mapping(indicator_variables_mapping_buffer.data(), constraint.arity());
            constraint.function()->add_to_lp(ilp_data, constraint_indicator_variables_mapping, ilp_constraint_builder_buffer);
        }



        // pass to highs
        highs_model_.lp_.num_col_ = ilp_data.col_lower().size();
        highs_model_.lp_.num_row_ = ilp_data.row_lower().size();
        highs_model_.lp_.col_cost_ = std::move(ilp_data.col_cost());
        highs_model_.lp_.col_lower_ = std::move(ilp_data.col_lower());
        highs_model_.lp_.col_upper_ = std::move(ilp_data.col_upper());
        highs_model_.lp_.row_lower_ = std::move(ilp_data.row_lower());
        highs_model_.lp_.row_upper_ = std::move(ilp_data.row_upper());

        // Here the orientation of the matrix is column-wise
        highs_model_.lp_.a_matrix_.format_ = MatrixFormat::kRowwise;
        // a_start_ has num_col_1 entries, and the last entry is the number
        // of nonzeros in A, allowing the number of nonzeros in the last
        // column to be defined
        highs_model_.lp_.a_matrix_.start_ = std::move(ilp_data.astart());
        highs_model_.lp_.a_matrix_.index_ = std::move(ilp_data.aindex());
        highs_model_.lp_.a_matrix_.value_  = std::move(ilp_data.avalue());

        highs_model_.lp_.a_matrix_.start_.push_back(highs_model_.lp_.a_matrix_.index_.size());
    }
    

    void IlpHighs::optimize(
        reporter_callback_wrapper_type & reporter_callback,
        repair_callback_wrapper_type & /*repair_callback not used*/
    ) {
        

        reporter_callback.begin();
        
        const bool continue_opt =  reporter_callback.report();



        // Create a Highs instance
        Highs highs;
        highs.setOptionValue("log_to_console", false);
        HighsStatus return_status;
        //
        // Pass the model to HiGHS
        return_status = highs.passModel(highs_model_);
        assert(return_status==HighsStatus::kOk);
        // If a user passes a model with entries in
        // model.lp_.a_matrix_.value_ less than (the option)
        // small_matrix_value in magnitude, they will be ignored. A logging
        // message will indicate this, and passModel will return
        // HighsStatus::kWarning
        //
        // Get a const reference to the LP data in HiGHS
        const HighsLp& lp = highs.getLp();
        //
        // Solve the model
        return_status = highs.run();
        const HighsModelStatus& model_status = highs.getModelStatus();
        //std::cout << "Model status: " << highs.modelStatusToString(model_status) << std::endl;
        assert(return_status==HighsStatus::kOk);
        //
        // Get the model status
        assert(model_status==HighsModelStatus::kOptimal);

        // Get the solution information
        const HighsInfo& info = highs.getInfo();
        // std::cout << "Simplex iteration count: " << info.simplex_iteration_count << std::endl;
        // std::cout << "Objective function value: " << info.objective_function_value << std::endl;
        // std::cout << "Primal  solution status: " << highs.solutionStatusToString(info.primal_solution_status) << std::endl;
        // std::cout << "Dual    solution status: " << highs.solutionStatusToString(info.dual_solution_status) << std::endl;
        // std::cout << "Basis: " << highs.basisValidityToString(info.basis_validity) << std::endl;
        const bool has_values = info.primal_solution_status;
        const bool has_duals = info.dual_solution_status;
        const bool has_basis = info.basis_validity;
        
        // Get the solution values and basis
        const HighsSolution& solution = highs.getSolution();
        const HighsBasis& basis = highs.getBasis();

        // Now indicate that all the variables must take integer values
        highs_model_.lp_.integrality_.resize(lp.num_col_);
        for (int col=0; col < lp.num_col_; col++)
            highs_model_.lp_.integrality_[col] = HighsVarType::kInteger;

        highs.passModel(highs_model_);
        // Solve the model
        return_status = highs.run();
        assert(return_status==HighsStatus::kOk);


    
        for(std::size_t vi=0; vi< this->model().space().size(); ++vi)
        {
            for(discrete_label_type l=0; l<this->model().space()[vi]; ++l)
            {
                const auto lp_var = indicator_variable_mapping_[vi] + l;
                const auto lp_sol = solution.col_value[lp_var];
                if ( solution.col_value[lp_var] > 0.5)
                {
                    best_solution_[vi] = l;
                    break;
                }
            }
        }
        
        this->best_sol_value_ = this->model()(this->best_solution_, false);





        
        reporter_callback.end();
    }

    SolutionValue IlpHighs::best_solution_value() const{
        return this->best_sol_value_;
    }
    SolutionValue IlpHighs::current_solution_value() const{
        return this->current_sol_value_;
    }

    const typename IlpHighs::solution_type & IlpHighs::best_solution()const{
        return this->best_solution_;
    }
    const typename IlpHighs::solution_type & IlpHighs::current_solution()const{
        return this->current_solution_;
    }





} // namespace nxtgm
