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
  private:
    using QpboImpl = QPBO<double>;
    using ImplProbeOptions = typename QpboImpl::ProbeOptions;

  public:
    using BaseProbeOptions = typename QpboBase::ProbeOptions;
    ~QpboKolmogorov() = default;
    QpboKolmogorov(std::size_t num_nodes, std::size_t num_edges);

    void add_unary_term(std::size_t node, const double *cost) override;
    void add_pairwise_term(std::size_t node1, std::size_t node2, const double *cost) override;
    void solve() override;

    void stitch() override;
    int get_region(std::size_t node) override;
    void merge_parallel_edges() override;
    void get_regions(std::size_t *regions) override;
    void compute_weak_persistencies() override;
    void improve(int N, int *order_array, int *fixed_nodes = nullptr) override;
    bool improve() override;
    bool lower_bound() override;
    double compute_energy(int options) override;
    void probe(int *mapping, const BaseProbeOptions &options) override;
    int get_label(std::size_t node) const override;
    void set_label(std::size_t node, int label) override;
    void get_labels(int *labeling) const override;
    void set_labels(const int *labeling) override;
    void merge_mappings(int node_num, int *mapping1, int *mapping2) override;

  private:
    std::size_t num_nodes_;
    std::size_t num_edges_;

    std::unique_ptr<QpboImpl> qpbo_;
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
    qpbo_ = std::make_unique<QpboImpl>(num_nodes_, num_edges_);
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
void QpboKolmogorov::solve()
{
    qpbo_->Solve();
}

void QpboKolmogorov::stitch()
{
    qpbo_->Stitch();
}
int QpboKolmogorov::get_region(std::size_t node)
{
    return qpbo_->GetRegion(node);
}
void QpboKolmogorov::merge_parallel_edges()
{
    qpbo_->MergeParallelEdges();
}
void QpboKolmogorov::get_regions(std::size_t *regions)
{
    for (std::size_t i = 0; i < num_nodes_; ++i)
    {
        regions[i] = qpbo_->GetRegion(i);
    }
}
void QpboKolmogorov::compute_weak_persistencies()
{
    qpbo_->ComputeWeakPersistencies();
}
void QpboKolmogorov::improve(int N, int *order_array, int *fixed_nodes)
{
    qpbo_->Improve(N, order_array, fixed_nodes);
}
bool QpboKolmogorov::improve()
{
    return qpbo_->Improve();
}
bool QpboKolmogorov::lower_bound()
{
    return qpbo_->ComputeTwiceLowerBound() / 2.0;
}
double QpboKolmogorov::compute_energy(int options)
{
    return qpbo_->ComputeTwiceEnergy(options) / 2.0;
}
void QpboKolmogorov::probe(int *mapping, const typename QpboBase::ProbeOptions &options)
{
    ImplProbeOptions impl_options;
    impl_options.directed_constraints = static_cast<int>(options.directed_constraints);
    impl_options.weak_persistencies = static_cast<int>(options.persistencies);
    impl_options.C = options.C;
    impl_options.order_array = options.order_array;
    impl_options.order_seed = options.order_seed;
    impl_options.dilation = options.dilation;
    impl_options.callback_fn = options.callback_fn;

    qpbo_->Probe(mapping, impl_options);
}
int QpboKolmogorov::get_label(std::size_t node) const
{
    return qpbo_->GetLabel(node);
}
void QpboKolmogorov::set_label(std::size_t node, int label)
{
    qpbo_->SetLabel(node, label);
}
void QpboKolmogorov::get_labels(int *labeling) const
{
    for (std::size_t i = 0; i < num_nodes_; ++i)
    {
        labeling[i] = qpbo_->GetLabel(i);
    }
}
void QpboKolmogorov::set_labels(const int *labeling)
{
    for (std::size_t i = 0; i < num_nodes_; ++i)
    {
        qpbo_->SetLabel(i, labeling[i]);
    }
}
void QpboKolmogorov::merge_mappings(int node_num, int *mapping1, int *mapping2)
{
    QpboImpl::MergeMappings(node_num, mapping1, mapping2);
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
