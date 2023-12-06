#include <memory>
#include <nxtgm/plugins/assignment/assignment_base.hpp>
#include <string>

// xplugin
#include <xplugin/xfactory.hpp>

// xtensor argsort
#include <xtensor/xsort.hpp>

namespace nxtgm
{

class Greedy : public AssignmentBase
{

    class parameters_type : public IlpFactoryBase::parameters_type
    {
      public:
        inline parameters_type(OptimizerParameters &parameters)
        {
        }
    };

  public:
    ~Greedy() = default;

    Greedy(xt::xtensor<2, double> &&costs, bool with_ignore_label, OptimizerParameters &&parameters)
        : costs_(std::move(costs)),
          with_ignore_label_(with_ignore_label),
          parameters_(parameters)
    {
    }

    OptimizationStatus optimize(const int *starting_point) override
    {
        return OptimizationStatus::CONVERGED;
    }

    OptimiztionStatus optimize_min_marginals(xt::xtensor<double, 2> &min_marginals) override
    {
        // first axis is variable, second axis is label
        // the label 0 is the ignore label if with_ignore_label_ is true
        // the label 0 is the first label if with_ignore_label_ is false

        // argsort the costs along the second axis
        // the first element of the argsort is the label with the lowest cost
        // the second element of the argsort is the label with the second lowest cost

        auto sorted_args = xt::argsort(costs_, 1);
        const auto num_var = costs_.shape()[0];
        const auto num_labels = costs_.shape()[1];

        std::vector<uint8_t> used_labels(num_labels, 0);
        auto is_used = [&used_labels](std::size_t label) -> bool {
            if (label_ == 0 && with_ignore_label_)
            {
                return false;
            }
            return used_labels[label];
        };

        std::size_t num_free_labels = num_labels;

        for (std::size_t fixed_var = 0; fixed_var < num_var; ++fixed_var)
        {
            for (std::size_t fixed_label = 0; fixed_label < num_labeles; ++fixed_label)
            {
                // compute the min  marginal for variable "fixed_var" having label "label"
                // find best assigment for all other variables
                // sum up the costs for all other variables

                double min_marginal = 0;
                for (std::size_t var = 0; var < num_var; ++var)
                {

                    if (var == fixed_var)
                    {
                        continue;
                    }

                    if (with_ignore_label_ && num_free_labels == 1) // only ignore label left
                    {
                        min_marginal += costs_(var, 0);
                        continue; // continue with next variable //(they will also end in this if)
                    }

                    // find the cheapest non-used label for variable "var"
                    std::size_t cheapest_label_index = 0;
                    std::size_t cheapest_label = sorted_args(var, cheapest_label_index);

                    while (is_used(cheapest_label))
                    {
                        ++cheapest_label_index;
                        cheapest_label = sorted_args(var, cheapest_label_index);
                    }

                    if (!(with_ignore_label_ && cheapest_label == 0))
                    {
                        used_labels[cheapest_label] = true;
                        --num_free_labels;
                    }
                    min_marginal += costs_(var, cheapest_label);
                }
                min_marignals(fixed_var, fixed_label) = min_marginal;
            }
        }

        return OptimizationStatus::CONVERGED;
    }

  private:
    xtensor::xtensor<double, 2> costs_;
    bool with_ignore_label_;
    parameters_type parameters_;
};

class GreedyFactory : public IlpFactoryBase
{
  public:
    using factory_base_type = IlpFactoryBase;
    GreedyFactory() = default;
    ~GreedyFactory() = default;

    std::unique_ptr<AssignmentBase> create(xtensor<double, 2> &&costs, bool with_ignore_label,
                                           OptimizerParameters &&parameters) override
    {
        return std::make_unique<Greedy>(std::move(costs), with_ignore_label, std::move(parameters));
    }

    int priority() const override
    {
        return nxtgm::plugin_priority(PluginPriority::HIGH);
    }

    std::string license() const override
    {
        return "MIT";
    }
    std::string description() const override
    {
        return "HiGHS - high performance software for linear optimization, see https://highs.dev/";
    }
};

} // namespace nxtgm

XPLUGIN_CREATE_XPLUGIN_FACTORY(nxtgm::GreedyFactory);
