#pragma once
#include <cstddef>
#include <memory>
#include <string>

#include <nxtgm/models/gm/discrete_gm/gm.hpp>
#include <nxtgm/nxtgm.hpp>
#include <nxtgm/optimizers/optimizer_parameters.hpp>
#include <nxtgm/plugins/plugin_priority.hpp>

namespace nxtgm
{

enum class ProposalConsumerStatus
{
    ACCEPTED,
    REJECTED,
    EXIT
};

class ProposalGenBase
{
  public:
    ProposalGenBase() = default;
    virtual ~ProposalGenBase() = default;

    virtual void generate(const discrete_label_type *best, discrete_label_type *proposal,
                          std::function<ProposalConsumerStatus()> consumer) = 0;
};

class ProposalGenFactoryBase
{
  public:
    virtual ~ProposalGenFactoryBase() = default;

    inline static std::string plugin_type()
    {
        return "proposal_gen";
    }

    inline static std::string plugin_dir_env_var()
    {
        return "NXTGM_PROPOSAL_GEN_PLUGIN_PATH";
    }

    // create an instance of the plugin
    virtual std::unique_ptr<ProposalGenBase> create(const DiscreteGm &gm, OptimizerParameters &&parameters) = 0;

    // priority of the plugin (higher means more important)
    virtual int priority() const = 0;

    // license of the plugin
    virtual std::string license() const = 0;

    // description of the plugin
    virtual std::string description() const = 0;
};

} // namespace nxtgm
