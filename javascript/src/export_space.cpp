#include <emscripten/bind.h>
#include <nxtgm/spaces/discrete_space.hpp>

namespace nxtgm
{

namespace em = emscripten;

void export_space()
{
    em::class_<DiscreteSpace>("DiscreteSpace").constructor<std::size_t, std::size_t>();
}

} // namespace nxtgm
