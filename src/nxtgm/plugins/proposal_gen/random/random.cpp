#include <memory>
#include <nxtgm/plugins/proposal_gen/proposal_gen_base.hpp>
#include <random>
#include <string>

// xplugin
#include <xplugin/xfactory.hpp>

namespace nxtgm
{

class ProposalGenRandom : public ProposalGenBase
{

    class parameters_type
    {
      public:
        inline parameters_type(OptimizerParameters &parameters)
        {
            parameters.assign_and_pop("num_iterations", num_iterations);
            parameters.assign_and_pop("exit_after_n_rejections", exit_after_n_rejections);
            parameters.assign_and_pop("p_flip", p_flip);
            parameters.assign_and_pop("seed", seed);
        }

        std::size_t num_iterations = 1000;
        std::size_t exit_after_n_rejections = 100;
        float p_flip = 0.05;
        std::size_t seed = 42;
    };

  public:
    ~ProposalGenRandom() = default;
    ProposalGenRandom(const DiscreteGm &gm, OptimizerParameters &&parameters)
        : ProposalGenBase(),
          gm_(gm),
          parameters_(parameters),
          max_num_labels_(gm_.space().max_num_labels())
    {
        ensure_all_handled("ProposalGenRandom", parameters);
    }

    void generate(const discrete_label_type *best, discrete_label_type *proposal,
                  std::function<ProposalConsumerStatus()> consumer) override
    {

        std::mt19937 gen(parameters_.seed);
        std::size_t n_rejections = 0;

        // dist for coin flip
        std::bernoulli_distribution flip_dist(parameters_.p_flip);

        for (std::size_t iter = 0; iter < parameters_.num_iterations; ++iter)
        {
            for (std::size_t i = 0; i < gm_.num_variables(); ++i)
            {

                // flip?
                if (flip_dist(gen))
                {
                    const auto num_labels = gm_.num_labels(i);
                    std::uniform_int_distribution<> dis(0, num_labels - 1);
                    proposal[i] = dis(gen);
                }
                else
                {
                    proposal[i] = best[i];
                }
            }
            // std::cout<<std::endl;

            auto status = consumer();
            if (status == ProposalConsumerStatus::EXIT)
            {
                return;
            }
            else if (status != ProposalConsumerStatus::ACCEPTED)
            {
                n_rejections = 0;
            }
            else
            {
                ++n_rejections;
                if (n_rejections >= parameters_.exit_after_n_rejections)
                {
                    return;
                }
            }
        }
    }

  private:
    const DiscreteGm &gm_;
    parameters_type parameters_;
    discrete_label_type max_num_labels_;
};

class ProposalGenRandomFactory : public ProposalGenFactoryBase
{
  public:
    using factory_base_type = ProposalGenFactoryBase;
    ProposalGenRandomFactory() = default;
    ~ProposalGenRandomFactory() = default;

    std::unique_ptr<ProposalGenBase> create(const DiscreteGm &gm, OptimizerParameters &&parameters) override
    {
        return std::make_unique<ProposalGenRandom>(gm, std::move(parameters));
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
        return "Random proposal generator for matching problems -- ie with a unique label constraint";
    }
};

} // namespace nxtgm

XPLUGIN_CREATE_XPLUGIN_FACTORY(nxtgm::ProposalGenRandomFactory);
