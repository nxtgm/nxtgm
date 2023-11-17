#include <nxtgm/nxtgm.hpp>
#include <nxtgm/plugins/proposal_gen/proposal_gen_base.hpp>

namespace nxtgm
{

plugin_registry<ProposalGenFactoryBase> &ProposalGenFactoryBase::get_registry()
{
    static plugin_registry<ProposalGenFactoryBase> registry;
    return registry;
}

} // namespace nxtgm
