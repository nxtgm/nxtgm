#include <memory>
#include <nxtgm/plugins/proposal_gen/proposal_gen_base.hpp>
#include <string>

// xplugin
#include <xplugin/xfactory.hpp>

namespace nxtgm
{

class ProposalGenTesting : public ProposalGenBase
{

    class parameters_type
    {
      public:
        inline parameters_type(OptimizerParameters &parameters)
        {
            parameters.assign_and_pop("name", name);
        }

        std::string name = "icm";
    };

  public:
    ~ProposalGenTesting() = default;
    ProposalGenTesting(const DiscreteGm &gm, OptimizerParameters &&parameters)
        : ProposalGenBase(),
          gm_(gm),
          parameters_(parameters)
    {
        ensure_all_handled("ProposalGenTesting", parameters);
    }

    void generate(const discrete_label_type *best, discrete_label_type *proposal,
                  std::function<ProposalConsumerStatus()> consumer) override
    {

        std::copy(best, best + gm_.num_variables(), proposal);

        bool changes = true;
        while (changes)
        {
            changes = false;
            for (std::size_t vi = 0; vi < gm_.num_variables(); ++vi)
            {

                const auto num_labels = gm_.num_labels(vi);
                const auto current_label = best[vi];

                for (discrete_label_type proposed_label = 0; proposed_label < num_labels; ++proposed_label)
                {
                    if (proposed_label == current_label)
                    {
                        continue;
                    }
                    proposal[vi] = proposed_label;
                    auto status = consumer();
                    if (status == ProposalConsumerStatus::EXIT)
                    {
                        return;
                    }
                    else if (status == ProposalConsumerStatus::ACCEPTED)
                    {
                        changes = true;
                        if (best[vi] != proposal[vi])
                        {
                            throw std::runtime_error("ProposalGenTesting: accepted but not changed");
                        }
                    }
                    else
                    {
                        proposal[vi] = best[vi];
                    }
                }
            }
        }
    }

  private:
    const DiscreteGm &gm_;
    parameters_type parameters_;
};

class ProposalGenTestingFactory : public ProposalGenFactoryBase
{
  public:
    using factory_base_type = ProposalGenFactoryBase;
    ProposalGenTestingFactory() = default;
    ~ProposalGenTestingFactory() = default;

    std::unique_ptr<ProposalGenBase> create(const DiscreteGm &gm, OptimizerParameters &&parameters) override
    {
        return std::make_unique<ProposalGenTesting>(gm, std::move(parameters));
    }

    int priority() const override
    {
        return nxtgm::plugin_priority(PluginPriority::VERY_LOW);
    }

    std::string license() const override
    {
        return "MIT";
    }
    std::string description() const override
    {
        return "proposal generators used for testing";
    }
};

} // namespace nxtgm

XPLUGIN_CREATE_XPLUGIN_FACTORY(nxtgm::ProposalGenTestingFactory);
