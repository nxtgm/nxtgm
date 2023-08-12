#include <nxtgm/plugins/min_st_cut_base.hpp>

#include <memory>
#include <string>

#include <graph.h>

// xplugin
#include <xplugin/xfactory.hpp>

#include <iostream>

namespace nxtgm
{

class MinStCutKolmogorov : public MinStCutBase
{
  public:
    ~MinStCutKolmogorov() = default;
    MinStCutKolmogorov(std::size_t num_nodes, std::size_t num_edges);

    void add_terminal_weights(std::size_t node, double cap_source, double cap_sink) override;
    void add_edge(std::size_t node1, std::size_t node2, double cap, double rev_cap) override;
    double solve(discrete_label_type *solution) override;

  private:
    using max_flow_graph_type = Graph<double, double, double>;

    std::size_t num_nodes_;
    std::size_t num_edges_;

    std::unique_ptr<max_flow_graph_type> max_flow_;
};

class MinStCutKolmogorovFactory : public MinStCutFactoryBase
{
  public:
    using factory_base_type = MinStCutFactoryBase;
    ~MinStCutKolmogorovFactory() = default;
    std::unique_ptr<MinStCutBase> create(std::size_t num_nodes, std::size_t num_edges) override;
    int priority() const override;
    std::string license() const override;
    std::string description() const override;
};

// constructor
MinStCutKolmogorov::MinStCutKolmogorov(std::size_t num_nodes, std::size_t num_edges)
    : num_nodes_(num_nodes),
      num_edges_(num_edges)
{
    max_flow_.reset(new max_flow_graph_type(num_nodes_, num_edges_));
    max_flow_->add_node(num_nodes_);
}

void MinStCutKolmogorov::add_terminal_weights(std::size_t node, double cap_source, double cap_sink)
{
    max_flow_->add_tweights(node, cap_source, cap_sink);
}
void MinStCutKolmogorov::add_edge(std::size_t node1, std::size_t node2, double cap, double rev_cap)
{
    max_flow_->add_edge(node1, node2, cap, rev_cap);
}
double MinStCutKolmogorov::solve(discrete_label_type *solution)
{
    const double flow = max_flow_->maxflow();
    for (std::size_t i = 0; i < num_nodes_; ++i)
    {
        solution[i] = max_flow_->what_segment(i) == max_flow_graph_type::SOURCE ? 0 : 1;
    }
    return flow;
}

std::unique_ptr<MinStCutBase> MinStCutKolmogorovFactory::create(std::size_t num_nodes, std::size_t num_edges)
{
    return std::make_unique<MinStCutKolmogorov>(num_nodes, num_edges);
}

std::string MinStCutKolmogorovFactory::license() const
{
    return "GPL";
}
std::string MinStCutKolmogorovFactory::description() const
{
    return "MinStCut bsed on MaxFlow from Kolmogorov";
}

int MinStCutKolmogorovFactory::priority() const
{
    return plugin_priority(PluginPriority::HIGH);
}
} // namespace nxtgm

XPLUGIN_CREATE_XPLUGIN_FACTORY(nxtgm::MinStCutKolmogorovFactory);
