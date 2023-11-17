#include <nxtgm/nxtgm.hpp>
#include <nxtgm/plugins/min_st_cut/min_st_cut_base.hpp>

namespace nxtgm
{

plugin_registry<MinStCutFactoryBase> &MinStCutFactoryBase::get_registry()
{
    static plugin_registry<MinStCutFactoryBase> registry;
    return registry;
}

} // namespace nxtgm
