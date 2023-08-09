#ifndef NXTGM_PLUGINS_QPBO_BASE_HPP
#define NXTGM_PLUGINS_QPBO_BASE_HPP

#include <cstddef>
#include <memory>
#include <string>

#include <nxtgm/plugins/plugin.hpp>

namespace nxtgm
{
class MinStCutBase
{
  public:
    virtual ~MinStCutBase() = default;

    virtual void add_terminal_weights(std::size_t node, double cap_source, double cap_sink) = 0;
    virtual void add_edge(std::size_t node1, std::size_t node2, double cap, double rev_cap) = 0;
    virtual double solve(int *uint8_t) = 0;
};

class MinStCutFactoryBase
{
  public:
    static std::string plugin_type()
    {
        return "min_st_cut";
    }

    static std::string plugin_dir_env_var()
    {
        return "NXTGM_MIN_ST_CUT_PLUGIN_PATH";
    }

    virtual ~MinStCutFactoryBase() = default;

    // create an instance of the plugin
    virtual std::unique_ptr<MinStCutBase> create(std::size_t num_nodes, std::size_t num_edges) = 0;

    // priority of the plugin (higher means more important)
    virtual int priority() const = 0;

    // license of the plugin
    virtual std::string license() const = 0;

    // description of the plugin
    virtual std::string description() const = 0;
};

} // namespace nxtgm

#endif // NXTGM_PLUGINS_QPBO_BASE_HPP
