#ifndef NXTGM_PLUGINS_MIN_ST_CUT_BASE_HPP
#define NXTGM_PLUGINS_MIN_ST_CUT_BASE_HPP

#include <cstddef>
#include <memory>

namespace nxtgm
{
class MinStCutBase
{
  public:
    virtual ~MinStCutBase() = default;

    virtual void add_unary_term(std::size_t node, const double *cost) = 0;
    virtual void add_pairwise_term(std::size_t node1, std::size_t node2, const double *cost) = 0;
    virtual void solve(int *labels) = 0;
};

class MinStCutFactoryBase
{
  public:
    virtual ~MinStCutFactoryBase() = default;

    // create an instance of the plugin
    virtual std::unique_ptr<MinStCutBase> create(std::size_t num_nodes, std::size_t num_edges) = 0;

    // license of the plugin
    virtual std::string license() const = 0;

    // description of the plugin
    virtual std::string description() const = 0;
};

} // namespace nxtgm

#endif // NXTGM_PLUGINS_MIN_ST_CUT_BASE_HPP
