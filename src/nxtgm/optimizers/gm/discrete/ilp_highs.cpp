
#include <nxtgm/optimizers/gm/discrete/ilp_highs.hpp>
#include <nxtgm/utils/n_nested_loops.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>

#include <nxtgm/utils/lp.hpp>

namespace nxtgm
{

OptimizationStatus
highsModelStatusToOptimizationStatus(Highs& highs,
                                     HighsModelStatus model_status)
{
    if (model_status == HighsModelStatus::kOptimal)
    {
        return OptimizationStatus::OPTIMAL;
    }
    else if (model_status == HighsModelStatus::kModelError ||
             model_status == HighsModelStatus::kSolveError)
    {
        throw std::runtime_error("Error in Highs:" +
                                 highs.modelStatusToString(model_status));
        return OptimizationStatus::UNKNOWN;
    }
    else if (model_status == HighsModelStatus::kInfeasible ||
             model_status == HighsModelStatus::kUnboundedOrInfeasible)
    {
        return OptimizationStatus::INFEASIBLE;
    }
    else if (model_status == HighsModelStatus::kUnbounded)
    {
        throw std::runtime_error(
            "nxgtm internal error: unbounded Model in Highs:" +
            highs.modelStatusToString(model_status));
        return OptimizationStatus::UNKNOWN;
    }
    else if (model_status == HighsModelStatus::kObjectiveBound)
    {
        throw std::runtime_error(
            "nxgtm internal error: kObjectiveBound is unexpected status:" +
            highs.modelStatusToString(model_status));
        return OptimizationStatus::UNKNOWN;
    }
    else if (model_status == HighsModelStatus::kObjectiveTarget)
    {
        throw std::runtime_error(
            "nxgtm internal error: kObjectiveBound is unexpected status:" +
            highs.modelStatusToString(model_status));
        return OptimizationStatus::UNKNOWN;
    }
    else if (model_status == HighsModelStatus::kTimeLimit)
    {
        return OptimizationStatus::TIME_LIMIT_REACHED;
    }
    else if (model_status == HighsModelStatus::kIterationLimit)
    {
        return OptimizationStatus::UNKNOWN;
    }
    else
    {
        throw std::runtime_error("nxgtm internal error: unexpected status:" +
                                 highs.modelStatusToString(model_status));
        return OptimizationStatus::UNKNOWN;
    }
}

IlpHighs::IlpHighs(const DiscreteGm& gm, const parameters_type& parameters,
                   const solution_type& initial_solution)
    : base_type(gm), parameters_(parameters), best_solution_(),
      current_solution_(), best_sol_value_(), current_sol_value_(),
      lower_bound_(-std::numeric_limits<energy_type>::infinity()), ilp_data_(),
      indicator_variable_mapping_(gm.space()), highs_model_()
{
    if (initial_solution.empty())
    {
        best_solution_ = solution_type(gm.space().size());
    }
    else
    {
        best_solution_ = initial_solution;
    }
    best_sol_value_ = this->model().evaluate(best_solution_, false);
    current_solution_ = best_solution_;
    current_sol_value_ = best_sol_value_;

    this->setup_lp();
}

void IlpHighs::setup_lp()
{

    // shortcuts
    const auto& model = this->model();
    const auto& space = model.space();

    // add inter variables for all the indicator variables
    // (objective will be added later)
    ilp_data_.add_variables(
        indicator_variable_mapping_.num_indicator_variables(), 0.0, 1.0, 0.0,
        true);

    // sum to one constraints
    for (std::size_t vi = 0; vi < space.size(); ++vi)
    {
        ilp_data_.begin_constraint(1.0, 1.0);
        for (discrete_label_type l = 0; l < space[vi]; ++l)
        {
            ilp_data_.add_constraint_coefficient(
                indicator_variable_mapping_[vi] + l, 1.0);
        }
    }

    // add all the factors to the ilp
    IlpFactorBuilderBuffer ilp_factor_builder_buffer;
    std::vector<std::size_t> indicator_variables_mapping_buffer(
        model.max_arity());

    for (auto&& factor : model.factors())
    {
        factor.map_from_model(indicator_variable_mapping_,
                              indicator_variables_mapping_buffer);
        factor.function()->add_to_lp(ilp_data_,
                                     indicator_variables_mapping_buffer.data(),
                                     ilp_factor_builder_buffer);
    };

    // add constraints to the ilp
    IlpConstraintBuilderBuffer ilp_constraint_builder_buffer;
    for (auto&& constraint : model.constraints())
    {
        constraint.map_from_model(indicator_variable_mapping_,
                                  indicator_variables_mapping_buffer);
        constraint.function()->add_to_lp(
            ilp_data_, indicator_variables_mapping_buffer.data(),
            ilp_constraint_builder_buffer);
    }

    // pass to highs
    highs_model_.lp_.num_col_ = ilp_data_.col_lower().size();
    highs_model_.lp_.num_row_ = ilp_data_.row_lower().size();
    highs_model_.lp_.col_cost_ = std::move(ilp_data_.col_cost());
    highs_model_.lp_.col_lower_ = std::move(ilp_data_.col_lower());
    highs_model_.lp_.col_upper_ = std::move(ilp_data_.col_upper());
    highs_model_.lp_.row_lower_ = std::move(ilp_data_.row_lower());
    highs_model_.lp_.row_upper_ = std::move(ilp_data_.row_upper());
    highs_model_.lp_.a_matrix_.format_ = MatrixFormat::kRowwise;
    highs_model_.lp_.a_matrix_.start_ = std::move(ilp_data_.astart());
    highs_model_.lp_.a_matrix_.index_ = std::move(ilp_data_.aindex());
    highs_model_.lp_.a_matrix_.value_ = std::move(ilp_data_.avalue());
    highs_model_.lp_.a_matrix_.start_.push_back(
        highs_model_.lp_.a_matrix_.index_.size());
}

OptimizationStatus
IlpHighs::optimize(reporter_callback_wrapper_type& reporter_callback,
                   repair_callback_wrapper_type& /*repair_callback not used*/,
                   const_discrete_solution_span)
{

    reporter_callback.begin();

    const bool continue_opt = reporter_callback.report();

    // Create a Highs instance
    Highs highs;
    highs.setOptionValue("log_to_console", parameters_.highs_log_to_console);
    HighsStatus return_status;

    // Pass the model to HiGHS
    return_status = highs.passModel(highs_model_);
    assert(return_status == HighsStatus::kOk);

    // Get a const reference to the LP data in HiGHS
    const HighsLp& lp = highs.getLp();

    // Solve the model
    return_status = highs.run();
    OptimizationStatus status =
        highsModelStatusToOptimizationStatus(highs, highs.getModelStatus());
    if (status == OptimizationStatus::INFEASIBLE)
    {
        reporter_callback.end();
        return OptimizationStatus::INFEASIBLE;
    }

    // get the lower bound
    lower_bound_ = highs.getHighsInfo().objective_function_value;
    reporter_callback.report();

    // Get the model status
    // assert(model_status == HighsModelStatus::kOptimal);

    // Get the solution information
    const HighsInfo& info = highs.getInfo();
    // std::cout << "Simplex iteration count: " << info.simplex_iteration_count
    // << std::endl; std::cout << "Objective function value: " <<
    // info.objective_function_value << std::endl; std::cout << "Primal solution
    // status: " << highs.solutionStatusToString(info.primal_solution_status) <<
    // std::endl; std::cout << "Dual    solution status: " <<
    // highs.solutionStatusToString(info.dual_solution_status) << std::endl;
    // std::cout << "Basis: " <<
    // highs.basisValidityToString(info.basis_validity)
    // << std::endl;
    const bool has_values = info.primal_solution_status;
    const bool has_duals = info.dual_solution_status;
    const bool has_basis = info.basis_validity;

    // Get the solution values and basis
    const HighsSolution& solution = highs.getSolution();
    const HighsBasis& basis = highs.getBasis();

    if (parameters_.integer)
    {
        highs_model_.lp_.integrality_.resize(lp.num_col_);
        for (int col = 0; col < lp.num_col_; col++)
        {
            highs_model_.lp_.integrality_[col] =
                ilp_data_.is_integer()[col] ? HighsVarType::kInteger
                                            : HighsVarType::kContinuous;
        }
        highs.passModel(highs_model_);
        return_status = highs.run();
        assert(return_status == HighsStatus::kOk);

        status =
            highsModelStatusToOptimizationStatus(highs, highs.getModelStatus());
        if (status == OptimizationStatus::INFEASIBLE)
        {
            reporter_callback.end();
            return OptimizationStatus::INFEASIBLE;
        }
    }

    const bool all_integral =
        indicator_variable_mapping_.lp_solution_to_model_solution(
            solution.col_value, best_solution_);

    this->best_sol_value_ = this->model().evaluate(this->best_solution_);
    reporter_callback.end();

    return status;
}

energy_type IlpHighs::lower_bound() const { return this->lower_bound_; }

SolutionValue IlpHighs::best_solution_value() const
{
    return this->best_sol_value_;
}
SolutionValue IlpHighs::current_solution_value() const
{
    return this->current_sol_value_;
}

const typename IlpHighs::solution_type& IlpHighs::best_solution() const
{
    return this->best_solution_;
}
const typename IlpHighs::solution_type& IlpHighs::current_solution() const
{
    return this->current_solution_;
}

} // namespace nxtgm
