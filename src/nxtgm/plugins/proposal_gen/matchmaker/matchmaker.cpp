#include <memory>
#include <nxtgm/plugins/proposal_gen/proposal_gen_base.hpp>
#include <string>

// xplugin
#include <xplugin/xfactory.hpp>

namespace nxtgm
{

class ProposalGenMatchMaker : public ProposalGenBase
{

    class parameters_type
    {
      public:
        inline parameters_type(OptimizerParameters &parameters)
        {
        }
    };

  public:
    ~ProposalGenMatchMaker() = default;
    ProposalGenMatchMaker(const DiscreteGm &gm, OptimizerParameters &&parameters)
        : ProposalGenBase(),
          gm_(gm),
          parameters_(parameters),
          max_num_labels_(gm_.space().max_num_labels())
    {
        ensure_all_handled("ProposalGenMatchMaker", parameters);
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

                for (int increment : {-1, 1})
                {
                    for (std::size_t i = 0; i < gm_.num_variables(); ++i)
                    {
                        const auto num_labels = gm_.num_labels(i);
                        auto new_label = static_cast<int>(best[i]) + increment;
                        if (new_label < 0)
                        {
                            new_label = num_labels - 1;
                        }
                        else if (new_label >= num_labels)
                        {
                            new_label = 0;
                        }
                        proposal[i] = new_label;
                    }
                    // std::cout<<"best    :";
                    // for (std::size_t i = 0; i < gm_.num_variables(); ++i)
                    // {
                    //     std::cout<<best[i]<<" ";
                    // }
                    // std::cout<<std::endl;
                    // std::cout<<"proposig:";
                    // for (std::size_t i = 0; i < gm_.num_variables(); ++i)
                    // {
                    //     std::cout<<proposal[i]<<" ";
                    // }
                    // std::cout<<std::endl;
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
    }

  private:
    const DiscreteGm &gm_;
    parameters_type parameters_;
    discrete_label_type max_num_labels_;
};

class ProposalGenMatchMakerFactory : public ProposalGenFactoryBase
{
  public:
    using factory_base_type = ProposalGenFactoryBase;
    ProposalGenMatchMakerFactory() = default;
    ~ProposalGenMatchMakerFactory() = default;

    std::unique_ptr<ProposalGenBase> create(const DiscreteGm &gm, OptimizerParameters &&parameters) override
    {
        return std::make_unique<ProposalGenMatchMaker>(gm, std::move(parameters));
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
        return "matchmaker proposal generator for matching problems -- ie with a unique label constraint";
    }
};

} // namespace nxtgm

XPLUGIN_CREATE_XPLUGIN_FACTORY(nxtgm::ProposalGenMatchMakerFactory);
