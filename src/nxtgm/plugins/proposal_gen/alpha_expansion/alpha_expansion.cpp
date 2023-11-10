#include <memory>
#include <nxtgm/plugins/proposal_gen/proposal_gen_base.hpp>
#include <string>

// xplugin
#include <xplugin/xfactory.hpp>

namespace nxtgm
{

class ProposalGenAlphaExpansion : public ProposalGenBase
{

    class parameters_type
    {
      public:
        inline parameters_type(OptimizerParameters &parameters)
        {
            parameters.assign_and_pop("starting_alpha", starting_alpha);
        }

        std::size_t starting_alpha = 0;
    };

  public:
    ~ProposalGenAlphaExpansion() = default;
    ProposalGenAlphaExpansion(const DiscreteGm &gm, OptimizerParameters &&parameters)
        : ProposalGenBase(),
          gm_(gm),
          parameters_(parameters),
          max_num_labels_(gm_.space().max_num_labels()),
          current_alpha_(parameters_.starting_alpha)
    {
        ensure_all_handled("ProposalGenAlphaExpansion", parameters);
    }

    void generate(const discrete_label_type *best, discrete_label_type *proposal,
                  std::function<ProposalConsumerStatus()> consumer) override
    {

        bool changes = true;
        while (changes)
        {
            changes = false;
            for (discrete_label_type alpha = 0; alpha < max_num_labels_; ++alpha)
            {
                for (std::size_t i = 0; i < gm_.num_variables(); ++i)
                {
                    const auto num_labels = gm_.num_labels(i);
                    proposal[i] = alpha < num_labels ? alpha : num_labels - 1;
                }
                auto status = consumer();
                if (status == ProposalConsumerStatus::EXIT)
                {
                    return;
                }
                else if (status == ProposalConsumerStatus::ACCEPTED)
                {
                    changes = true;
                }
            }
        }
    }

  private:
    const DiscreteGm &gm_;
    parameters_type parameters_;
    discrete_label_type max_num_labels_;
    discrete_label_type current_alpha_;
};

class ProposalGenAlphaExpansionFactory : public ProposalGenFactoryBase
{
  public:
    using factory_base_type = ProposalGenFactoryBase;
    ProposalGenAlphaExpansionFactory() = default;
    ~ProposalGenAlphaExpansionFactory() = default;

    std::unique_ptr<ProposalGenBase> create(const DiscreteGm &gm, OptimizerParameters &&parameters) override
    {
        return std::make_unique<ProposalGenAlphaExpansion>(gm, std::move(parameters));
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
        return "alpha expansion like proposal generator";
    }
};

} // namespace nxtgm

XPLUGIN_CREATE_XPLUGIN_FACTORY(nxtgm::ProposalGenAlphaExpansionFactory);
