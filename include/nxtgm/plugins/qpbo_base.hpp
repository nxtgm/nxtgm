#ifndef NXTGM_PLUGINS_QPBO_BASE_HPP
#define NXTGM_PLUGINS_QPBO_BASE_HPP

#include <cstddef>
#include <memory>
#include <string>

#include <nxtgm/plugins/plugin.hpp>

namespace nxtgm
{
class QpboBase
{
  public:
    virtual ~QpboBase() = default;

    virtual void add_unary_term(std::size_t node, const double *cost) = 0;
    virtual void add_pairwise_term(std::size_t node1, std::size_t node2, const double *cost) = 0;
    virtual void solve(int *labels) = 0;
};

class QpboFactoryBase
{
  public:
    static std::string plugin_type()
    {
        return "qpbo";
    }

    static std::string plugin_dir_env_var()
    {
        return "NXTGM_QPBO_PLUGIN_PATH";
    }

    virtual ~QpboFactoryBase() = default;

    // create an instance of the plugin
    virtual std::unique_ptr<QpboBase> create(std::size_t num_nodes, std::size_t num_edges) = 0;

    // priority of the plugin (higher means more important)
    virtual int priority() const = 0;

    // license of the plugin
    virtual std::string license() const = 0;

    // description of the plugin
    virtual std::string description() const = 0;
};

} // namespace nxtgm

#endif // NXTGM_PLUGINS_QPBO_BASE_HPP
