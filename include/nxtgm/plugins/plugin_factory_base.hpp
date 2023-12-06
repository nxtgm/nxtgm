#pragma once
#include <memory>
#include <string>

namespace nxtgm
{

class PluginFactoryBase
{
  public:
    virtual ~PluginFactoryBase() = default;

    // priority of the plugin (higher means more important)
    virtual int priority() const = 0;

    // license of the plugin
    virtual std::string license() const = 0;

    // description of the plugin
    virtual std::string description() const = 0;
};

} // namespace nxtgm
