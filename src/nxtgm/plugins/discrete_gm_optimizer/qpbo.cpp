#include <iostream>
#include <limits>
#include <nxtgm/nxtgm.hpp>
#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>

#include <nxtgm/plugins/plugin_registry.hpp>
#include <nxtgm/plugins/qpbo/qpbo_base.hpp>
#include <nxtgm/utils/timer.hpp>

#include <algorithm>
#include <random>

namespace nxtgm
{

class Qpbo : public DiscreteGmOptimizerBase
{
    class parameters_type
    {
      public:
        inline parameters_type(OptimizerParameters &parameters)
        {
            if (auto it = parameters.string_parameters.find("qpbo_plugin_name");
                it != parameters.string_parameters.end())
            {
                qpbo_plugin_name = it->second;
                parameters.string_parameters.erase(it);
            }

            if (auto it = parameters.int_parameters.find("probing"); it != parameters.int_parameters.end())
            {
                probing = it->second;
                parameters.int_parameters.erase(it);
            }
            if (auto it = parameters.int_parameters.find("strong_persistencies"); it != parameters.int_parameters.end())
            {
                strong_persistencies = it->second;
                parameters.int_parameters.erase(it);
            }
            if (auto it = parameters.int_parameters.find("improving"); it != parameters.int_parameters.end())
            {
                improving = it->second;
                parameters.int_parameters.erase(it);
            }
            if (auto it = parameters.int_parameters.find("seed"); it != parameters.int_parameters.end())
            {
                seed = it->second;
                parameters.int_parameters.erase(it);
            }
        }

        std::string qpbo_plugin_name;
        bool probing = false;
        bool strong_persistencies = true;
        bool improving = false;
        unsigned seed = 0;
    };

  public:
    using base_type = DiscreteGmOptimizerBase;
    using solution_type = typename DiscreteGm::solution_type;

    using reporter_callback_wrapper_type = typename base_type::reporter_callback_wrapper_type;
    using repair_callback_wrapper_type = typename base_type::repair_callback_wrapper_type;

    inline static std::string name()
    {
        return "Qpbo";
    }
    virtual ~Qpbo() = default;

    Qpbo(const DiscreteGm &gm, OptimizerParameters &&parameters);

    OptimizationStatus optimize_impl(reporter_callback_wrapper_type &, repair_callback_wrapper_type &,
                                     const_discrete_solution_span starting_point) override;

    SolutionValue best_solution_value() const override;
    SolutionValue current_solution_value() const override;

    const solution_type &best_solution() const override;
    const solution_type &current_solution() const override;

    energy_type lower_bound() const override;

    bool is_partial_optimal(std::size_t variable_index) const override;

  private:
    parameters_type parameters_;
    SolutionValue best_solution_value_;
    solution_type best_solution_;
    energy_type lower_bound_;
    std::vector<int> qpbo_labels_;
    std::unique_ptr<QpboBase> qpbo_;
};

class QpboDiscreteGmOptimizerFactory : public DiscreteGmOptimizerFactoryBase
{
  public:
    using factory_base_type = DiscreteGmOptimizerFactoryBase;
    virtual ~QpboDiscreteGmOptimizerFactory() = default;
    std::unique_ptr<DiscreteGmOptimizerBase> create(const DiscreteGm &gm, OptimizerParameters &&params) const override
    {
        return std::make_unique<Qpbo>(gm, std::move(params));
    }
    int priority() const override
    {
        return plugin_priority(PluginPriority::HIGH);
    }
    std::string license() const override
    {
        return "MIT";
    }
    std::string description() const override
    {
        return "QPBO optimizer";
    }
    OptimizerFlags flags() const override
    {
        return OptimizerFlags::PartialOptimal | OptimizerFlags::OptimalOnBinarySecondOrderSubmodular |
               OptimizerFlags::OptimalOnTrees;
    }
};

} // namespace nxtgm

XPLUGIN_CREATE_XPLUGIN_FACTORY(nxtgm::QpboDiscreteGmOptimizerFactory);

namespace nxtgm
{
Qpbo::Qpbo(const DiscreteGm &gm, OptimizerParameters &&parameters)
    : base_type(gm, parameters),
      parameters_(parameters),
      best_solution_value_(),
      best_solution_(gm.num_variables(), 0),
      lower_bound_(std::numeric_limits<energy_type>::infinity()),
      qpbo_(nullptr),
      qpbo_labels_(gm.num_variables(), -1)
{
    ensure_all_handled(name(), parameters);

    best_solution_value_ = gm.evaluate(best_solution_, false /* early exit when infeasible*/);

    // check space
    if (gm.space().max_num_labels() != 2)
    {
        throw std::runtime_error("QPBO only supports binary variables");
    }
    // check for constraints
    if (gm.num_constraints() > 0)
    {
        throw std::runtime_error("QPBO does not support constraints");
    }
    // check arity
    if (gm.max_arity() > 2)
    {
        throw std::runtime_error("QPBO only supports pairwise factors");
    }

    // build the qpbo model count the number of edges
    size_t num_edges = 0;
    for (const auto &factor : gm.factors())
    {
        if (const auto arity = factor.arity(); arity == 2)
            num_edges += 1;
    }
    auto factory = get_plugin_registry<QpboFactoryBase>().get_factory(parameters_.qpbo_plugin_name);
    qpbo_ = factory->create(gm.num_variables(), num_edges);

    double energies[4] = {0, 0, 0, 0};
    for (size_t i = 0; i < gm.num_factors(); ++i)
    {
        const auto &factor = gm.factor(i);
        if (const auto arity = factor.arity(); arity == 1)
        {
            factor.copy_energies(energies);
            qpbo_->add_unary_term(factor.variables()[0], energies);
        }
        else if (arity == 2)
        {
            factor.copy_energies(energies);
            qpbo_->add_pairwise_term(factor.variables()[0], factor.variables()[1], energies);
        }
    }
}

energy_type Qpbo::lower_bound() const
{
    return lower_bound_;
}

OptimizationStatus Qpbo::optimize_impl(reporter_callback_wrapper_type &reporter_callback,
                                       repair_callback_wrapper_type & /*repair_callback not used*/,
                                       const_discrete_solution_span starting_point)
{
    // shortcut to the model
    const auto &gm = this->model();

    qpbo_->merge_parallel_edges();
    qpbo_->solve();
    if (!parameters_.strong_persistencies)
    {
        qpbo_->compute_weak_persistencies();
    }

    lower_bound_ = qpbo_->lower_bound();
    qpbo_->get_labels(qpbo_labels_.data());

    // count unlabeled variables
    std::size_t num_unlabeled = 0;
    for (std::size_t vi = 0; vi < gm.num_variables(); ++vi)
    {
        if (qpbo_labels_[vi] < 0)
        {
            ++num_unlabeled;
            best_solution_[vi] = 0;
        }
        else
        {
            best_solution_[vi] = qpbo_labels_[vi];
        }
    }

    if (num_unlabeled > 0 && (parameters_.probing || parameters_.improving))
    {
        std::vector<int> list_unlabeled(gm.num_variables());
        num_unlabeled = 0;
        for (std::size_t vi = 0; vi < gm.num_variables(); ++vi)
        {
            const auto qpbo_label = qpbo_labels_[vi];
            if (qpbo_label < 0)
            {
                ++num_unlabeled;
                best_solution_[vi] = 0;
                list_unlabeled[num_unlabeled - 1] = vi;
            }
            else
            {
                best_solution_[vi] = qpbo_label;
            }
        }

        // initialize mapping
        std::vector<int> mapping(gm.num_variables());
        for (std::size_t vi = 0; vi < gm.num_variables(); ++vi)
        {
            mapping[vi] = vi * 2;
        }

        if (parameters_.probing && num_unlabeled > 0)
        {
            typename QpboBase::ProbeOptions options;
            options.persistencies = QpboBase::Persistencies::Weak;

            std::vector<int> new_mapping(gm.num_variables());
            qpbo_->probe(new_mapping.data(), options);
            qpbo_->merge_mappings(gm.num_constraints(), mapping.data(), new_mapping.data());
            qpbo_->compute_weak_persistencies();

            // Read out entire labelling again (as weak persistencies may have changed)
            num_unlabeled = 0;
            for (std::size_t vi = 0; vi < gm.num_variables(); ++vi)
            {
                qpbo_labels_[vi] = qpbo_->get_label(mapping[vi] / 2);
                const auto qpbo_label = qpbo_labels_[vi];
                if (qpbo_label < 0)
                {
                    ++num_unlabeled;
                    best_solution_[vi] = 0;
                    list_unlabeled[num_unlabeled - 1] = vi;
                }
                else
                {
                    best_solution_[vi] = qpbo_label;
                }
            }
        }
        if (parameters_.improving && starting_point.size() == gm.num_variables())
        {
            std::vector<int> improve_order(gm.num_variables());

            // Set the labels to the starting point
            for (std::size_t i = 0; i < num_unlabeled; ++i)
            {
                improve_order[i] = mapping[list_unlabeled[i]] / 2;
                qpbo_->set_label(improve_order[i], starting_point[improve_order[i]]);
            }

            // randomize the order
            std::random_device rd;
            std::mt19937 g(parameters_.seed);
            std::shuffle(improve_order.begin(), improve_order.begin() + num_unlabeled, g);

            // QPBO-I
            qpbo_->improve();
            for (std::size_t i = 0; i < num_unlabeled; ++i)
            {
                best_solution_[list_unlabeled[i]] =
                    (qpbo_->get_label(mapping[list_unlabeled[i]] / 2) + mapping[list_unlabeled[i]]) % 2;
            }
        }
    }

    best_solution_value_ = gm.evaluate(best_solution_, false /* early exit when infeasible*/);

    return num_unlabeled == 0 ? OptimizationStatus::OPTIMAL : OptimizationStatus::CONVERGED;
}

SolutionValue Qpbo::best_solution_value() const
{
    return best_solution_value_;
}
SolutionValue Qpbo::current_solution_value() const
{
    return best_solution_value_;
}

const typename Qpbo::solution_type &Qpbo::best_solution() const
{
    return best_solution_;
}
const typename Qpbo::solution_type &Qpbo::current_solution() const
{
    return best_solution_;
}

bool Qpbo::is_partial_optimal(std::size_t variable_index) const
{
    return qpbo_labels_[variable_index] >= 0;
}

} // namespace nxtgm
