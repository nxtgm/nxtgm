#pragma once

#include <emscripten/bind.h>

#include <nxtgm/nxtgm.hpp>
#include <string>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

#include <nxtgm/models/gm/discrete_gm/gm.hpp>

#include <nxtgm/models/gm/discrete_gm/gm.hpp>
#include <nxtgm/plugins/proposal_gen/proposal_gen_base.hpp>

#include "convert.hpp"

namespace nxtgm
{

namespace em = emscripten;

struct JsConsumerProxy
{
    std::function<ProposalConsumerStatus()> consumer;
    // discrete_label_type *proposal;
    // discrete_label_type *best;
    // std::size_t size;
    ProposalConsumerStatus consume();
};

class JsProposalGen : public ProposalGenBase
{
  public:
    JsProposalGen(const DiscreteGm &gm_, em::val js_proposal_gen);

    void generate(const discrete_label_type *best, discrete_label_type *proposal,
                  std::function<ProposalConsumerStatus()> consumer) override;

  private:
    const DiscreteGm &gm_;
    em::val js_proposal_gen_;
};

class JsProposalGenFactory : public ProposalGenFactoryBase
{
  public:
    JsProposalGenFactory(em::val js_proposal_gen_factory);

    // create an instance of the plugin
    std::unique_ptr<ProposalGenBase> create(const DiscreteGm &gm, OptimizerParameters &&parameters) override;
    std::string license() const override;
    std::string description() const override;
    int priority() const override;

  private:
    em::val js_proposal_gen_factory_;
};

} // namespace nxtgm
