#include <nxtgm/nxtgm.hpp>
#include <nxtgm/plugins/assignment/assignment_base.hpp>

namespace nxtgm
{

plugin_registry<AssignmentFactoryBase> &AssignmentFactoryBase::get_registry()
{
    static plugin_registry<AssignmentFactoryBase> registry;
    return registry;
}

} // namespace nxtgm
