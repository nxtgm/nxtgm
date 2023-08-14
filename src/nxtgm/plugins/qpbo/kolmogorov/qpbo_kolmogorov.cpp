#include <nxtgm/plugins/qpbo/qpbo_base.hpp>

#include <memory>
#include <string>

// qpbo impl
#include <QPBO.h>

// xplugin
#include <xplugin/xfactory.hpp>

namespace nxtgm
{

class QpboKolmogorov : public QpboBase
{
  public:
    ~QpboKolmogorov() = default;
    QpboKolmogorov(std::size_t num_nodes, std::size_t num_edges);

    void add_unary_term(std::size_t node, const double *cost) override;
    void add_pairwise_term(std::size_t node1, std::size_t node2, const double *cost) override;
    void solve(int *labels) override;

  private:
    using qpbo_impl_type = QPBO<double>;

    std::size_t num_nodes_;
    std::size_t num_edges_;

    std::unique_ptr<qpbo_impl_type> qpbo_;
};

class QpboKolmogorovFactory : public QpboFactoryBase
{
  public:
    using factory_base_type = QpboFactoryBase;
    ~QpboKolmogorovFactory() = default;
    std::unique_ptr<QpboBase> create(std::size_t num_nodes, std::size_t num_edges) override;
    int priority() const override;
    std::string license() const override;
    std::string description() const override;
};

// implementation of the plugin
QpboKolmogorov::QpboKolmogorov(std::size_t num_nodes, std::size_t num_edges)
    : num_nodes_(num_nodes),
      num_edges_(num_edges)
{
    qpbo_ = std::make_unique<qpbo_impl_type>(num_nodes_, num_edges_);
    qpbo_->AddNode(num_nodes_);
}
void QpboKolmogorov::add_unary_term(std::size_t node, const double *cost)
{
    qpbo_->AddUnaryTerm(node, cost[0], cost[1]);
}
void QpboKolmogorov::add_pairwise_term(std::size_t node1, std::size_t node2, const double *cost)
{
    qpbo_->AddPairwiseTerm(node1, node2, cost[0], cost[1], cost[2], cost[3]);
}
void QpboKolmogorov::solve(int *labels)
{
    qpbo_->MergeParallelEdges();
    qpbo_->Solve();
    qpbo_->ComputeWeakPersistencies();

    for (std::size_t i = 0; i < num_nodes_; ++i)
    {
        labels[i] = this->qpbo_->GetLabel(i);
    }
}

std::unique_ptr<QpboBase> QpboKolmogorovFactory::create(std::size_t num_nodes, std::size_t num_edges)
{
    return std::make_unique<QpboKolmogorov>(num_nodes, num_edges);
}
int QpboKolmogorovFactory::priority() const
{
    return nxtgm::plugin_priority(PluginPriority::HIGH);
}
std::string QpboKolmogorovFactory::license() const
{
    return "GPL";
}
std::string QpboKolmogorovFactory::description() const
{
    return "QPBO Kolmogorov";
}

} // namespace nxtgm

XPLUGIN_CREATE_XPLUGIN_FACTORY(nxtgm::QpboKolmogorovFactory);
