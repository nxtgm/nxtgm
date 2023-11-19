#include <memory>
#include <nxtgm/nxtgm.hpp>
#include <nxtgm/optimizers/gm/discrete/discrete_gm_optimizer_factory.hpp>
#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>
#include <nxtgm/plugins/proposal_gen/proposal_gen_base.hpp>

#include <string>

// xplugin
#include <xplugin/xfactory.hpp>

namespace nxtgm
{

class ProposalGenOptimizerBased : public ProposalGenBase
{
    class Callback : public ReporterCallbackBase<DiscreteGmOptimizerBase>
    {
      public:
        virtual ~Callback() = default;

        Callback(DiscreteGmOptimizerBase *optimizer, discrete_label_type *proposal, const discrete_label_type *best,
                 std::function<ProposalConsumerStatus()> consumer)
            : ReporterCallbackBase<DiscreteGmOptimizerBase>(optimizer),
              proposal_(proposal),
              best_(best),
              consumer_(consumer)
        {
            //
        }

        void begin() override
        {
            //
        }
        void end() override
        {
            consumer_();
        }
        bool report() override
        {
            const auto &current = this->optimizer()->current_solution();
            std::copy(current.begin(), current.end(), proposal_);
            auto consumer_status = consumer_();
            return consumer_status != ProposalConsumerStatus::EXIT;
        }

      private:
        discrete_label_type *proposal_;
        const discrete_label_type *best_;
        std::function<ProposalConsumerStatus()> consumer_;
    };

    class parameters_type
    {
      public:
        inline parameters_type(OptimizerParameters &parameters)
        {
            parameters.assign_and_pop("optimizer_name", optimizer_name);
            parameters.assign_and_pop("optimizer_parameters", optimizer_parameters);
        }

        std::string optimizer_name = "belief_propagation";
        OptimizerParameters optimizer_parameters;
    };

  public:
    ~ProposalGenOptimizerBased() = default;
    ProposalGenOptimizerBased(const DiscreteGm &gm, OptimizerParameters &&parameters)
        : ProposalGenBase(),
          gm_(gm),
          parameters_(parameters)
    {
        ensure_all_handled("ProposalGenOptimizerBased", parameters);
    }

    void generate(const discrete_label_type *best, discrete_label_type *proposal,
                  std::function<ProposalConsumerStatus()> consumer) override
    {

        auto expected_optimizer =
            discrete_gm_optimizer_factory(gm_, parameters_.optimizer_name, std::move(parameters_.optimizer_parameters));
        if (!expected_optimizer)
        {
            throw std::runtime_error(expected_optimizer.error());
        }
        auto optimizer = std::move(expected_optimizer.value());
        Callback callback(optimizer.get(), proposal, best, consumer);

        const_discrete_solution_span starting_point(proposal, gm_.num_variables());

        optimizer->optimize(&callback, nullptr, starting_point);
    }

  private:
    const DiscreteGm &gm_;
    parameters_type parameters_;
    discrete_label_type max_num_labels_;
    discrete_label_type current_alpha_;
};

class ProposalGenOptimizerBasedFactory : public ProposalGenFactoryBase
{
  public:
    using factory_base_type = ProposalGenFactoryBase;
    ProposalGenOptimizerBasedFactory() = default;
    ~ProposalGenOptimizerBasedFactory() = default;

    std::unique_ptr<ProposalGenBase> create(const DiscreteGm &gm, OptimizerParameters &&parameters) override
    {
        return std::make_unique<ProposalGenOptimizerBased>(gm, std::move(parameters));
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
        return "Higher order clique reduction by Alexander Fix";
    }
};

} // namespace nxtgm

XPLUGIN_CREATE_XPLUGIN_FACTORY(nxtgm::ProposalGenOptimizerBasedFactory);
