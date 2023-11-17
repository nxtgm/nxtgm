#include <nxtgm/nxtgm.hpp>
#include <nxtgm/plugins/ilp/ilp_base.hpp>

namespace nxtgm
{

plugin_registry<IlpFactoryBase> &IlpFactoryBase::get_registry()
{
    static plugin_registry<IlpFactoryBase> registry;
    return registry;
}

} // namespace nxtgm
