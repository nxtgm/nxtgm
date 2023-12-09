#include <nxtgm/nxtgm.hpp>
#include <nxtgm/optimizers/gm/discrete/fusion.hpp>
#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>
#include <nxtgm/plugins/plugin_registry.hpp>
#include <nxtgm/plugins/proposal_gen/proposal_gen_base.hpp>
#include <nxtgm/utils/timer.hpp>

namespace nxtgm
{

class FusionMoves;

class FusionMoves : public DiscreteGmOptimizerBase
{
    friend class FusionMovesProposalConsumer;

    class parameters_type
    {
      public:
        inline parameters_type(OptimizerParameters &&parameters)
        {
            parameters.assign_and_pop("fusion_parameters", fusion_parameters);
            parameters.assign_and_pop("proposal_gen_name", proposal_gen_name);
            parameters.assign_and_pop("proposal_gen_parameters", proposal_gen_parameters);
            parameters.assign_and_pop("max_iterations", max_iterations);
        }
        OptimizerParameters fusion_parameters;
        std::string proposal_gen_name = "alpha_expansion";
        OptimizerParameters proposal_gen_parameters;
        std::size_t max_iterations = 0; // 0 means the proposal generator decides
    };

  public:
    using base_type = DiscreteGmOptimizerBase;
    using solution_type = typename DiscreteGm::solution_type;

    using reporter_callback_wrapper_type = typename base_type::reporter_callback_wrapper_type;
    using repair_callback_wrapper_type = typename base_type::repair_callback_wrapper_type;

    inline static std::string name()
    {
        return "FusionMoves";
    }
    virtual ~FusionMoves() = default;

    FusionMoves(const DiscreteGm &gm, OptimizerParameters &&parameters);

    OptimizationStatus optimize_impl(reporter_callback_wrapper_type &, repair_callback_wrapper_type &,
                                     const_discrete_solution_span) override;

    SolutionValue best_solution_value() const override;
    SolutionValue current_solution_value() const override;

    const solution_type &best_solution() const override;
    const solution_type &current_solution() const override;

  private:
    parameters_type parameters_;
    solution_type best_solution_;
    mutable SolutionValue best_sol_value_;
    Fusion fusion_;
    std::unique_ptr<ProposalGenBase> proposal_gen_;
    std::size_t iteration_;
    mutable bool value_is_tidy_;
};

class FusionMovesFactory : public DiscreteGmOptimizerFactoryBase
{
  public:
    using factory_base_type = DiscreteGmOptimizerFactoryBase;
    virtual ~FusionMovesFactory() = default;
    expected<std::unique_ptr<DiscreteGmOptimizerBase>> create(const DiscreteGm &gm,
                                                              OptimizerParameters &&params) const override
    {
        return std::make_unique<FusionMoves>(gm, std::move(params));
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
        return "Fusion Moves";
    }
    OptimizerFlags flags() const override
    {
        return OptimizerFlags::WarmStartable;
    }
};
} // namespace nxtgm

XPLUGIN_CREATE_XPLUGIN_FACTORY(nxtgm::FusionMovesFactory);

namespace nxtgm
{

FusionMoves::FusionMoves(const DiscreteGm &gm, OptimizerParameters &&parameters)
    : base_type(gm, parameters),
      parameters_(std::move(parameters)),
      best_solution_(gm.num_variables()),
      best_sol_value_(gm.evaluate(best_solution_)),
      fusion_(gm, std::move(parameters_.fusion_parameters)),
      proposal_gen_(nullptr),
      iteration_(0),
      value_is_tidy_(false)

{
    ensure_all_handled(name(), parameters);

    auto factory = get_plugin_registry<ProposalGenFactoryBase>().get_factory(std::string("proposal_gen_") +
                                                                             parameters_.proposal_gen_name);

    proposal_gen_ = factory->create(gm, std::move(parameters_.proposal_gen_parameters));
}

OptimizationStatus FusionMoves::optimize_impl(reporter_callback_wrapper_type &reporter_callback,
                                              repair_callback_wrapper_type & /*repair_callback not used*/,
                                              const_discrete_solution_span starting_point)
{
    if (starting_point.size() > 0)
    {
        std::cout << "starting point size: " << starting_point.size() << std::endl;
        std::copy(starting_point.begin(), starting_point.end(), best_solution_.begin());
    }

    std::vector<discrete_label_type> proposal(this->model().num_variables());

    auto status = OptimizationStatus::CONVERGED;
    proposal_gen_->generate(best_solution_.data(), proposal.data(), [&]() {
        std::vector<discrete_label_type> fused(best_solution_.size());
        auto fuse_results = this->fusion_.fuse(best_solution_.data(), proposal.data(), fused.data());

        if (fuse_results != FusionResult::A)
        {
            value_is_tidy_ = false;
            std::copy(fused.begin(), fused.end(), best_solution_.begin());
        }

        ++iteration_;

        // reached max iterations?
        if (parameters_.max_iterations > 0 && iteration_ >= parameters_.max_iterations)
        {
            status = OptimizationStatus::ITERATION_LIMIT_REACHED;
            return ProposalConsumerStatus::EXIT;
        }

        // reached time limit?
        if (this->time_limit_reached())
        {
            status = OptimizationStatus::TIME_LIMIT_REACHED;
            return ProposalConsumerStatus::EXIT;
        }

        if (fuse_results != FusionResult::A)
        {
            // call visitor
            if (!this->report(reporter_callback))
            {
                status = OptimizationStatus::CALLBACK_EXIT;
                return ProposalConsumerStatus::EXIT;
            }

            return ProposalConsumerStatus::ACCEPTED;
        }
        else
        {
            return ProposalConsumerStatus::REJECTED;
        }
    });

    // shortcut to the model
    const auto &gm = this->model();

    return status;
}
SolutionValue FusionMoves::best_solution_value() const
{
    if (!value_is_tidy_)
    {
        best_sol_value_ = this->model().evaluate(this->best_solution_);
        value_is_tidy_ = true;
    }
    return this->best_sol_value_;
}
SolutionValue FusionMoves::current_solution_value() const
{
    return this->best_solution_value();
}

const typename FusionMoves::solution_type &FusionMoves::best_solution() const
{
    return this->best_solution_;
}
const typename FusionMoves::solution_type &FusionMoves::current_solution() const
{
    return this->best_solution_;
}

} // namespace nxtgm
