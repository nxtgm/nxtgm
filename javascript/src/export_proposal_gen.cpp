#include <emscripten/bind.h>

#include <nxtgm/nxtgm.hpp>
#include <string>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

#include <nxtgm/models/gm/discrete_gm/gm.hpp>

#include <nxtgm/models/gm/discrete_gm/gm.hpp>
#include <nxtgm/plugins/proposal_gen/proposal_gen_base.hpp>

#include "convert.hpp"
#include "proposal_gen.hpp"

namespace nxtgm
{

namespace em = emscripten;

std::shared_ptr<JsProposalGenFactory> _proposal_gen_smart_ptr_constructor(em::val js_proposal_gen_factory)
{
    return std::make_shared<JsProposalGenFactory>(js_proposal_gen_factory);
}

void export_proposal_gen()
{
    // enable shared ptrs
    em::class_<ProposalGenFactoryBase>("ProposalGenFactoryBase")
        .smart_ptr<std::shared_ptr<ProposalGenFactoryBase>>("ProposalGenFactoryBase");

    em::class_<JsProposalGen>("JsProposalGen");

    em::class_<JsProposalGenFactory, em::base<ProposalGenFactoryBase>>("JsProposalGenFactory")
        .smart_ptr_constructor("JsProposalGenFactory", &_proposal_gen_smart_ptr_constructor);

    em::enum_<ProposalConsumerStatus>("ProposalConsumerStatus")
        .value("ACCEPTED", ProposalConsumerStatus::ACCEPTED)
        .value("REJECTED", ProposalConsumerStatus::REJECTED)
        .value("EXIT", ProposalConsumerStatus::EXIT);

    em::class_<JsConsumerProxy>("JsConsumerProxy").function("consume", &JsConsumerProxy::consume);
}

} // namespace nxtgm
