#include <nxtgm/nxtgm.hpp>
#include <nxtgm/plugins/qpbo/qpbo_base.hpp>

namespace nxtgm
{

plugin_registry<QpboFactoryBase> &QpboFactoryBase::get_registry()
{
    static plugin_registry<QpboFactoryBase> registry;
    return registry;
}

} // namespace nxtgm
