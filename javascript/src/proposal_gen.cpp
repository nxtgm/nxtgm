#include "proposal_gen.hpp"

namespace nxtgm
{

ProposalConsumerStatus JsConsumerProxy::consume()
{
    return consumer();
}

JsProposalGen::JsProposalGen(const DiscreteGm &gm_, em::val js_proposal_gen)
    : gm_(gm_),
      js_proposal_gen_(js_proposal_gen)
{
}

void JsProposalGen::generate(const discrete_label_type *best, discrete_label_type *proposal,
                             std::function<ProposalConsumerStatus()> consumer)
{
    // create py::buffer_info from best
    // auto best_view = py::array(gm_.num_variables(), best, py::cast(*this, py::return_value_policy::reference));

    // best view as typed mem view
    auto best_view = em::val(em::typed_memory_view(gm_.num_variables(), best));
    auto proposal_view = em::val(em::typed_memory_view(gm_.num_variables(), proposal));

    JsConsumerProxy consumer_proxy;
    consumer_proxy.consumer = consumer;
    // consumer_proxy.size = gm_.num_variables();
    js_proposal_gen_.template call<void>("generate", consumer_proxy, best_view, proposal_view);
    // generate_function(consumer_proxy, best_view, proposal_view);
}

JsProposalGenFactory::JsProposalGenFactory(em::val js_proposal_gen_factory)
    : js_proposal_gen_factory_(js_proposal_gen_factory)
{
}

// create an instance of the plugin
std::unique_ptr<ProposalGenBase> JsProposalGenFactory::create(const DiscreteGm &gm, OptimizerParameters &&parameters)
{
    em::val js_proposal_gen = js_proposal_gen_factory_.template call<em::val>("create", em::val(gm));
    return std::make_unique<JsProposalGen>(gm, js_proposal_gen);
}

// irrelevant since this is not in the plugin registry
int JsProposalGenFactory::priority() const
{
    return 0;
}

// license of the plugin
std::string JsProposalGenFactory::license() const
{
    return js_proposal_gen_factory_["license"].as<std::string>();
}

// description of the plugin
std::string JsProposalGenFactory::description() const
{
    return js_proposal_gen_factory_["description"].as<std::string>();
}
} // namespace nxtgm
