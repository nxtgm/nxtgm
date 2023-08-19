#pragma once
#include <cstddef>
#include <memory>
#include <string>

#include <nxtgm/optimizers/optimizer_parameters.hpp>
#include <nxtgm/plugins/plugin.hpp>

namespace nxtgm
{

class QpboBase
{
  public:
    enum class Persistencies
    {
        Strong = 0,
        Weak = 1
    };

    enum class DirectedConstraints
    {
        OnlyExistingEdges = 0,
        AllPossileIfEnoughSpace = 1,
        AllPossible = 2
    };

    struct ProbeOptions
    {
        ProbeOptions() = default;
        ProbeOptions(const OptimizerParameters &parameters);

        DirectedConstraints directed_constraints = DirectedConstraints::AllPossible;
        Persistencies persistencies = Persistencies::Strong;
        double C;
        int *order_array = nullptr;
        unsigned int order_seed = 42;
        int dilation = 3;
        bool (*callback_fn)(int unlabeled_num) = nullptr;
    };

    virtual ~QpboBase() = default;

    virtual void add_unary_term(std::size_t node, const double *cost) = 0;
    virtual void add_pairwise_term(std::size_t node1, std::size_t node2, const double *cost) = 0;
    virtual void solve() = 0;

    // `
    virtual void stitch() = 0;
    virtual int get_region(std::size_t node) = 0;
    virtual void merge_parallel_edges() = 0;
    virtual void get_regions(std::size_t *regions) = 0;
    virtual void compute_weak_persistencies() = 0;
    virtual void improve(int N, int *order_array, int *fixed_nodes = nullptr) = 0;
    virtual bool improve() = 0;
    virtual bool lower_bound() = 0;
    virtual double compute_energy(int options) = 0;
    virtual void probe(int *mapping, const ProbeOptions &options) = 0;
    virtual int get_label(std::size_t node) const = 0;
    virtual void set_label(std::size_t node, int label) = 0;
    virtual void get_labels(int *labeling) const = 0;
    virtual void set_labels(const int *labeling) = 0;
    virtual void merge_mappings(int node_num, int *mapping1, int *mapping2) = 0;
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
