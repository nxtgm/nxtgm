#pragma once

namespace nxtgm
{

enum class PluginPriority : int
{
    VERY_LOW = 0,
    LOW = 1,
    MEDIUM = 2,
    HIGH = 3,
    VERY_HIGH = 4
};

inline int plugin_priority(PluginPriority p)
{
    return static_cast<int>(p);
}

} // namespace nxtgm
