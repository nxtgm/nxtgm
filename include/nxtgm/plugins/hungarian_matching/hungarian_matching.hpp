#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include <nxtgm/nxtgm.hpp>
#include <nxtgm/plugins/plugin.hpp>

namespace nxtgm
{
class HungarianMatchingBase
{
  public:
    virtual ~HungarianMatchingBase() = default;

    virtual void run(const xt::xtensor<double, 2> &costs, std::size_t *asignments) = 0;

    virtual void run(const xt::xtensor<double, 2> &costs, discrete_label_type *asignments,
                     std::size_t ignore_label // which label can be assigned multiple times
    )
    {
        const auto num_var = costs.shape()[0];
        const auto num_labels_plain = costs.shape()[1] - 1;

        // we need N extra labels to represent the ignore label
        const auto n_extra = num_var;

        xt::xtensor<double, 2> costs_extended({num_var, num_labels_plain + n_extra});
        for (std::size_t i = 0; i < num_var; ++i)
        {
            for (std::size_t j = 0; j < num_labels_plain; ++j)
            {
                costs_extended(i, j) = costs(i, j);
            }
            costs_extended(i, num_labels_plain + i) = costs(i, ignore_label);
        }

        // run the matching
        run(costs_extended, asignments);

        // remap the assignments
        for (std::size_t i = 0; i < num_var; ++i)
        {
            if (asignments[i] >= costs.shape()[1])
            {
                asignments[i] = ignore_label;
            }
        }
    }
};

class HungarianMatchingFactoryBase
{
  public:
    static std::string plugin_type()
    {
        return "hungarian_matching";
    }

    static std::string plugin_dir_env_var()
    {
        return "NXTGM_HUNGARIAN_MATCHING_PLUGIN_PATH";
    }

    virtual ~HungarianMatchingFactoryBase() = default;

    // create an instance of the plugin
    virtual std::unique_ptr<HungarianMatchingBase> create(std::size_t num_nodes, std::size_t num_edges) = 0;

    // priority of the plugin (higher means more important)
    virtual int priority() const = 0;

    // license of the plugin
    virtual std::string license() const = 0;

    // description of the plugin
    virtual std::string description() const = 0;
};

} // namespace nxtgm
