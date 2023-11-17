#include <memory>
#include <nxtgm/plugins/ilp/ilp_base.hpp>
#include <string>

// xplugin
#include <xplugin/xfactory.hpp>

// highs specific
#include <highs/Highs.h>

namespace nxtgm
{

OptimizationStatus highsModelStatusToOptimizationStatus(Highs &highs, HighsModelStatus model_status)
{
    if (model_status == HighsModelStatus::kOptimal)
    {
        return OptimizationStatus::OPTIMAL;
    }
    else if (model_status == HighsModelStatus::kModelError || model_status == HighsModelStatus::kSolveError)
    {
        throw std::runtime_error("Error in Highs:" + highs.modelStatusToString(model_status));
        return OptimizationStatus::UNKNOWN;
    }
    else if (model_status == HighsModelStatus::kInfeasible || model_status == HighsModelStatus::kUnboundedOrInfeasible)
    {
        return OptimizationStatus::INFEASIBLE;
    }
    else if (model_status == HighsModelStatus::kUnbounded)
    {
        throw std::runtime_error("nxgtm internal error: unbounded Model in Highs:" +
                                 highs.modelStatusToString(model_status));
        return OptimizationStatus::UNKNOWN;
    }
    else if (model_status == HighsModelStatus::kObjectiveBound)
    {
        throw std::runtime_error("nxgtm internal error: kObjectiveBound is unexpected status:" +
                                 highs.modelStatusToString(model_status));
        return OptimizationStatus::UNKNOWN;
    }
    else if (model_status == HighsModelStatus::kObjectiveTarget)
    {
        throw std::runtime_error("nxgtm internal error: kObjectiveBound is unexpected status:" +
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
        throw std::runtime_error("nxgtm internal error: unexpected status:" + highs.modelStatusToString(model_status));
        return OptimizationStatus::UNKNOWN;
    }
}

class IlpHighs : public IlpBase
{

    class parameters_type
    {
      public:
        inline parameters_type(OptimizerParameters &parameters)
        {
            if (auto it = parameters.int_parameters.find("time_limit_ms"); it != parameters.int_parameters.end())
            {
                time_limit_ = std::chrono::milliseconds(it->second);
                parameters.int_parameters.erase(it);
            }
        }
        std::chrono::duration<double> time_limit_ = std::chrono::duration<double>::max();
    };

  public:
    ~IlpHighs() = default;
    IlpHighs(OptimizerParameters &&parameters)
        : IlpBase(),
          parameters_(parameters),
          highs_(),
          highs_model_()
    {
        ensure_all_handled("IlpHighs", parameters);

        highs_.setOptionValue("time_limit", this->parameters_.time_limit_.count());
        highs_.setOptionValue("log_to_console", false);
    }

    std::size_t num_variables() const override
    {
        return highs_model_.lp_.num_col_;
    }

    IlpHighs(IlpData &&ilp_data, OptimizerParameters &&parameters)
        : IlpHighs(std::move(parameters))
    {
        // pass to highs
        highs_model_.lp_.num_col_ = ilp_data.col_lower().size();
        highs_model_.lp_.num_row_ = ilp_data.row_lower().size();
        highs_model_.lp_.col_cost_ = std::move(ilp_data.col_cost());
        highs_model_.lp_.col_lower_ = std::move(ilp_data.col_lower());
        highs_model_.lp_.col_upper_ = std::move(ilp_data.col_upper());
        highs_model_.lp_.row_lower_ = std::move(ilp_data.row_lower());
        highs_model_.lp_.row_upper_ = std::move(ilp_data.row_upper());
        highs_model_.lp_.a_matrix_.format_ = MatrixFormat::kRowwise;
        highs_model_.lp_.a_matrix_.start_ = std::move(ilp_data.astart());
        highs_model_.lp_.a_matrix_.index_ = std::move(ilp_data.aindex());
        highs_model_.lp_.a_matrix_.value_ = std::move(ilp_data.avalue());
        highs_model_.lp_.a_matrix_.start_.push_back(highs_model_.lp_.a_matrix_.index_.size());

        is_integer_ = std::move(ilp_data.is_integer());
    }

    OptimizationStatus optimize_lp() override
    {
        // std::cout << "optimize_lp 1" << std::endl;

        // Pass the model to HiGHS
        HighsStatus return_status = highs_.passModel(highs_model_);
        if (return_status != HighsStatus::kOk)
        {
            throw std::runtime_error(std::string("nxgtm ilpinternal error: failed to pass model to highs:") +
                                     highsStatusToString(return_status));
        }

        // // print the vec highs_model_.lp_.integrality_
        // for (std::size_t vi = 0; vi < highs_model_.lp_.integrality_.size(); ++vi)
        // {
        //     // std::cout << "var " << vi << " integrality: " << (
        //     // highs_model_.lp_.integrality_[vi]==HighsVarType::kInteger )<< std::endl;
        // }

        // std::cout << "optimize_lp 2" << std::endl;

        // Get a const reference to the LP data in HiGHS
        // const HighsLp &lp = highs_.getLp();

        // std::cout << "optimize_lp 3" << std::endl;

        // Solve the model
        return_status = highs_.run();
        // std::cout << "optimize_lp 4" << std::endl;
        OptimizationStatus status = highsModelStatusToOptimizationStatus(highs_, highs_.getModelStatus());
        // std::cout << "optimize_lp 5" << std::endl;
        if (status == OptimizationStatus::INFEASIBLE)
        {
            return OptimizationStatus::INFEASIBLE;
        }
        // std::cout << "optimize_lp 6" << std::endl;
        objective_value_ = highs_.getHighsInfo().objective_function_value;
        // std::cout << "optimize_lp 7" << std::endl;

        return status;
    }

    OptimizationStatus optimize_ilp() override
    {

        // not 100% sure if this is needed or what it does
        // but this infered from the example in the HiGHS repo
        if (const HighsInfo &info = highs_.getInfo(); !info.primal_solution_status)
        {
            throw std::runtime_error("nxgtm ilp highs internal error: !primal_solution_status");
        }
        else if (!info.dual_solution_status)
        {
            throw std::runtime_error("nxgtm ilp highs internal error: !dual_solution_status");
        }
        else if (!info.basis_validity)
        {
            throw std::runtime_error("nxgtm ilp highs internal error: !basis_validity");
        }

        // Get a const reference to the LP data in HiGHS
        const HighsLp &lp = highs_.getLp();

        highs_model_.lp_.integrality_.resize(lp.num_col_);
        for (int col = 0; col < lp.num_col_; col++)
        {
            highs_model_.lp_.integrality_[col] = is_integer_[col] ? HighsVarType::kInteger : HighsVarType::kContinuous;
        }
        highs_.passModel(highs_model_);
        HighsStatus return_status = highs_.run();
        OptimizationStatus status = highsModelStatusToOptimizationStatus(highs_, highs_.getModelStatus());
        if (status == OptimizationStatus::INFEASIBLE)
        {
            return OptimizationStatus::INFEASIBLE;
        }
        objective_value_ = highs_.getHighsInfo().objective_function_value;

        return status;
    }

    double get_objective_value() override
    {
        return objective_value_;
    }

    void get_solution(double *solution) override
    {
        const HighsLp &lp = highs_.getLp();
        const HighsSolution &solution_ = highs_.getSolution();
        for (int col = 0; col < lp.num_col_; col++)
        {
            solution[col] = solution_.col_value[col];
        }
    }

  private:
    parameters_type parameters_;
    Highs highs_;
    HighsModel highs_model_;
    std::vector<std::uint8_t> is_integer_;
    double objective_value_;
};

class IlpHighsFactory : public IlpFactoryBase
{
  public:
    using factory_base_type = IlpFactoryBase;
    IlpHighsFactory() = default;
    ~IlpHighsFactory() = default;

    std::unique_ptr<IlpBase> create(IlpData &&ilp_data, OptimizerParameters &&parameters) override
    {
        return std::make_unique<IlpHighs>(std::move(ilp_data), std::move(parameters));
    }

    int priority() const override
    {
        return nxtgm::plugin_priority(PluginPriority::HIGH);
    }

    std::string license() const override
    {
        return "MIT";
    }
    std::string description() const override
    {
        return "HiGHS - high performance software for linear optimization, see https://highs.dev/";
    }
};

} // namespace nxtgm

XPLUGIN_CREATE_XPLUGIN_FACTORY(nxtgm::IlpHighsFactory);
