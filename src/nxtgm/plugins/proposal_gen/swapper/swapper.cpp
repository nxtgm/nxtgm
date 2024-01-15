#include <memory>
#include <nxtgm/plugins/proposal_gen/proposal_gen_base.hpp>
#include <random>
#include <string>

// xplugin
#include <xplugin/xfactory.hpp>

namespace nxtgm
{

class Swapper : public ProposalGenBase
{

    class parameters_type
    {
      public:
        inline parameters_type(OptimizerParameters &parameters)
        {
            parameters.assign_and_pop("num_iterations", num_iterations);
            parameters.assign_and_pop("exit_after_n_rejections", exit_after_n_rejections);
            parameters.assign_and_pop("n_swap", n_swap);
            parameters.assign_and_pop("seed", seed);
        }

        std::size_t num_iterations = 1000;
        std::size_t exit_after_n_rejections = 100;
        float n_swap = 0.2;
        std::size_t seed = 42;
    };

  public:
    ~Swapper() = default;
    Swapper(const DiscreteGm &gm, OptimizerParameters &&parameters)
        : ProposalGenBase(),
          gm_(gm),
          parameters_(parameters),
          max_num_labels_(gm_.space().max_num_labels())
    {
        ensure_all_handled("Swapper", parameters);

        // ensure space is simple
        if (!gm_.space().is_simple())
        {
            throw std::runtime_error("Swapper only works with simple spaces");
        }
    }

    void generate(const discrete_label_type *best, discrete_label_type *proposal,
                  std::function<ProposalConsumerStatus()> consumer) override
    {

        std::mt19937 gen(parameters_.seed);
        std::size_t n_rejections = 0;

        // dist to get a random variable
        std::uniform_int_distribution<std::size_t> var_dist(0, gm_.num_variables() - 1);

        std::size_t flip_n = parameters_.n_swap > 1.0 ? std::floor(parameters_.n_swap)
                                                      : std::floor(parameters_.n_swap * gm_.num_variables());

        for (std::size_t iter = 0; iter < parameters_.num_iterations; ++iter)
        {
            std::copy(best, best + gm_.num_variables(), proposal);
            bool any = false;
            for (std::size_t i = 0; i < flip_n; ++i)
            {
                // get a random variable
                const auto var_0 = var_dist(gen);
                const auto var_1 = var_dist(gen);

                if (var_0 == var_1 || best[var_0] == best[var_1])
                {
                    continue;
                }
                std::swap(proposal[var_0], proposal[var_1]);
                any = true;
            }
            if (!any)
            {
                continue;
            }
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

class SwapperFactory : public ProposalGenFactoryBase
{
  public:
    using factory_base_type = ProposalGenFactoryBase;
    SwapperFactory() = default;
    ~SwapperFactory() = default;

    std::unique_ptr<ProposalGenBase> create(const DiscreteGm &gm, OptimizerParameters &&parameters) override
    {
        return std::make_unique<Swapper>(gm, std::move(parameters));
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

XPLUGIN_CREATE_XPLUGIN_FACTORY(nxtgm::SwapperFactory);
