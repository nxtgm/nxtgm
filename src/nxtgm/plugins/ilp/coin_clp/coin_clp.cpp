#include <memory>
#include <nxtgm/plugins/ilp/ilp_base.hpp>
#include <string>

// xplugin
#include <xplugin/xfactory.hpp>

// coin-(clp) specific
#include "ClpSimplex.hpp"
#include "CoinBuild.hpp"
#include "CoinHelperFunctions.hpp"
#include "CoinModel.hpp"
#include "CoinTime.hpp"

// warm start
#include <CoinWarmStartVector.hpp>

// For Branch and bound
#include "CbcModel.hpp"
#include "OsiSolverInterface.hpp"
// #include "CbcBranchUser.hpp"
//  #include "CbcCompareUser.hpp"
#include "CbcCutGenerator.hpp"
#include "CbcHeuristicLocal.hpp"
#include "OsiClpSolverInterface.hpp"

// Cuts

#include "CglClique.hpp"
#include "CglFlowCover.hpp"
#include "CglGomory.hpp"
#include "CglKnapsackCover.hpp"
#include "CglMixedIntegerRounding.hpp"
#include "CglOddHole.hpp"
#include "CglProbing.hpp"

// Heuristics

#include "CbcHeuristic.hpp"

namespace nxtgm
{

class coin_parameters_type : public IlpFactoryBase::parameters_type
{
  public:
    inline coin_parameters_type(OptimizerParameters &parameters)
        : IlpFactoryBase::parameters_type(parameters)
    {
        ensure_all_handled("coin_parameters_type", parameters);
    }
};

template <typename MODEL>
void coin_model_from_ilp_data(IlpData &ilp_data, MODEL &model)
{
    // pass to highs
    const auto num_col = ilp_data.col_lower().size();
    const auto num_row = ilp_data.row_lower().size();
    const auto num_elements = ilp_data.aindex().size();

    ilp_data.astart().push_back(num_elements);

    CoinPackedMatrix byRow(false, num_col, num_row, num_elements, ilp_data.avalue().data(), ilp_data.aindex().data(),
                           ilp_data.astart().data(), NULL);

    model.loadProblem(byRow, ilp_data.col_lower().data(), ilp_data.col_upper().data(), ilp_data.col_cost().data(),
                      ilp_data.row_lower().data(), ilp_data.row_upper().data());
}

class CoinClp : public IlpBase
{
    using parameters_type = coin_parameters_type;

  public:
    ~CoinClp() = default;
    CoinClp(parameters_type &&parameters)
        : IlpBase(),
          parameters_(std::move(parameters))
    {
        // log level
        osi_clp_lp_solver_.setLogLevel(parameters_.log_level);

        // time limit
        osi_lp_solver_ = &osi_clp_lp_solver_;

        // get the model
        osi_clp_lp_solver_.getModelPtr()->setMaximumSeconds(parameters_.time_limit.count());
    }

    std::size_t num_variables() const override
    {
        return osi_clp_lp_solver_.getNumCols();
    }

    CoinClp(IlpData &&ilp_data, parameters_type &&parameters)
        : CoinClp(std::move(parameters))
    {
        coin_model_from_ilp_data(ilp_data, osi_clp_lp_solver_);
    }

    OptimizationStatus optimize(const double *starting_point) override
    {
        if (starting_point != nullptr)
        {
            // copy to owned memory
            double *warm = new double[num_variables()];
            std::copy(starting_point, starting_point + num_variables(), warm);

            CoinWarmStartVector<double> ws(num_variables(), warm);
            osi_lp_solver_->setWarmStart(&ws);
        }
        OptimizationStatus status = OptimizationStatus::OPTIMAL;
        osi_lp_solver_->initialSolve();
        return status;
    }

    double get_objective_value() override
    {
        return osi_lp_solver_->getObjValue();
    }

    void get_solution(double *solution) override
    {
        const double *columnPrimal = osi_lp_solver_->getColSolution();
        std::copy(columnPrimal, columnPrimal + num_variables(), solution);
    }

  private:
    parameters_type parameters_;
    OsiSolverInterface *osi_lp_solver_;
    OsiClpSolverInterface osi_clp_lp_solver_;
};

class CoinCbc : public IlpBase
{
    using parameters_type = coin_parameters_type;

  public:
    ~CoinCbc() = default;
    CoinCbc(parameters_type &&parameters)
        : IlpBase(),
          parameters_(std::move(parameters)),
          cbc_model_(nullptr)
    {
    }

    std::size_t num_variables() const override
    {
        return cbc_model_->getNumCols();
    }

    CoinCbc(IlpData &&ilp_data, parameters_type &&parameters)
        : CoinCbc(std::move(parameters))
    {
        // the lp model
        coin_model_from_ilp_data(ilp_data, osi_clp_lp_solver_);
        osi_lp_solver_ = &osi_clp_lp_solver_;

        const auto &is_integer = ilp_data.is_integer();
        for (std::size_t i = 0; i < is_integer.size(); ++i)
        {
            if (is_integer[i])
            {
                osi_lp_solver_->setInteger(i);
            }
        }

        cbc_model_ = std::make_unique<CbcModel>(*osi_lp_solver_);

        // log level
        osi_clp_lp_solver_.setLogLevel(parameters_.log_level);
        cbc_model_->solver()->messageHandler()->setLogLevel(parameters_.log_level);
        cbc_model_->setLogLevel(parameters_.log_level);
        if (parameters_.log_level <= 1)
        {
            cbc_model_->solver()->setHintParam(OsiDoReducePrint, true, OsiHintTry);
        }

        // time limit
        cbc_model_->setMaximumSeconds(parameters_.time_limit.count());
    }

    OptimizationStatus optimize(const double *starting_point) override
    {

        if (starting_point != nullptr)
        {
            // copy to owned memory
            double *warm = new double[num_variables()];
            std::copy(starting_point, starting_point + num_variables(), warm);

            CoinWarmStartVector<double> ws(num_variables(), warm);
            osi_lp_solver_->setWarmStart(&ws);
        }

        OptimizationStatus status = OptimizationStatus::OPTIMAL;

        CglProbing generator1;
        generator1.setUsingObjective(true);
        generator1.setMaxPass(3);
        generator1.setMaxProbe(100);
        generator1.setMaxLook(50);
        generator1.setRowCuts(3);
        //  generator1.snapshot(*model.solver());
        // generator1.createCliques(*model.solver(),2,1000,true);
        // generator1.setMode(0);

        CglGomory generator2;
        // try larger limit
        generator2.setLimit(300);

        CglKnapsackCover generator3;

        CglOddHole generator4;
        generator4.setMinimumViolation(0.005);
        generator4.setMinimumViolationPer(0.00002);
        // try larger limit
        generator4.setMaximumEntries(200);

        CglClique generator5;
        generator5.setStarCliqueReport(false);
        generator5.setRowCliqueReport(false);

        CglMixedIntegerRounding mixedGen;
        CglFlowCover flowGen;

        // Add in generators
        cbc_model_->addCutGenerator(&generator1, -1, "Probing");
        cbc_model_->addCutGenerator(&generator2, -1, "Gomory");
        cbc_model_->addCutGenerator(&generator3, -1, "Knapsack");
        cbc_model_->addCutGenerator(&generator4, -1, "OddHole");
        cbc_model_->addCutGenerator(&generator5, -1, "Clique");
        cbc_model_->addCutGenerator(&flowGen, -1, "FlowCover");
        cbc_model_->addCutGenerator(&mixedGen, -1, "MixedIntegerRounding");

        cbc_model_->solver()->messageHandler()->setLogLevel(parameters_.log_level);

        // Allow rounding heuristic

        // CbcRounding heuristic1(*cbc_model_);
        // cbc_model_->addHeuristic(&heuristic1);

        // And local search when new solution found

        // CbcHeuristicLocal heuristic2(*cbc_model_);
        // cbc_model_->addHeuristic(&heuristic2);

        // solve continuously
        cbc_model_->initialSolve();

        // // Could tune more
        // cbc_model_->setMinimumDrop(CoinMin(1.0,
        // std::fabs(cbc_model_->getMinimizationObjValue()) * 1.0e-3 + 1.0e-4));

        // if (cbc_model_->getNumCols() < 500)
        // {
        //     cbc_model_->setMaximumCutPassesAtRoot(-100); // always do 100 if possible
        // }
        // else if (cbc_model_->getNumCols() < 5000)
        // {
        //     cbc_model_->setMaximumCutPassesAtRoot(100); // use minimum drop
        // }
        // else
        // {
        //     cbc_model_->setMaximumCutPassesAtRoot(20);
        // }

        if (cbc_model_->getNumCols() < 5000)
        {
            cbc_model_->setNumberStrong(10);
        }
        cbc_model_->solver()->setIntParam(OsiMaxNumIterationHotStart, 100);

        cbc_model_->branchAndBound();

        return status;
    }

    double get_objective_value() override
    {
        return cbc_model_->getMinimizationObjValue();
    }

    void get_solution(double *solution) override
    {
        const double *sol = cbc_model_->solver()->getColSolution();
        std::copy(sol, sol + num_variables(), solution);
    }

  private:
    parameters_type parameters_;
    OsiSolverInterface *osi_lp_solver_;
    OsiClpSolverInterface osi_clp_lp_solver_;
    std::unique_ptr<CbcModel> cbc_model_;
};

class CoinClpFactory : public IlpFactoryBase
{
  public:
    using factory_base_type = IlpFactoryBase;
    CoinClpFactory() = default;
    ~CoinClpFactory() = default;

    std::unique_ptr<IlpBase> create(IlpData &&ilp_data, OptimizerParameters &&parameters) override
    {
        coin_parameters_type coin_parameters(parameters);

        if (coin_parameters.integer)
        {
            return std::make_unique<CoinCbc>(std::move(ilp_data), std::move(coin_parameters));
        }
        else
        {
            return std::make_unique<CoinClp>(std::move(ilp_data), std::move(coin_parameters));
        }
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
        return "coin clp / cbc";
    }
};

} // namespace nxtgm

XPLUGIN_CREATE_XPLUGIN_FACTORY(nxtgm::CoinClpFactory);
