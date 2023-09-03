#include <iostream>
#include <limits>
#include <nxtgm/nxtgm.hpp>
#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>

#include <nxtgm/plugins/hocr/hocr_base.hpp>
#include <nxtgm/plugins/plugin_registry.hpp>
#include <nxtgm/plugins/qpbo/qpbo_base.hpp>
#include <nxtgm/utils/timer.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>

namespace nxtgm
{

class Hqpbo : public DiscreteGmOptimizerBase
{
  public:
    static constexpr std::size_t max_arity = 10;
    class parameters_type
    {
      public:
        inline parameters_type(OptimizerParameters &parameters)
        {
            if (auto it = parameters.string_parameters.find("hocr_plugin_name");
                it != parameters.string_parameters.end())
            {
                hocr_plugin_name = it->second;
                parameters.string_parameters.erase(it);
            }

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
        std::string hocr_plugin_name;
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
        return "Hqpbo";
    }
    virtual ~Hqpbo() = default;

    Hqpbo(const DiscreteGm &gm, OptimizerParameters &&parameters);

    OptimizationStatus optimize_impl(reporter_callback_wrapper_type &, repair_callback_wrapper_type &,
                                     const_discrete_solution_span starting_point) override;

    SolutionValue best_solution_value() const override;
    SolutionValue current_solution_value() const override;

    const solution_type &best_solution() const override;
    const solution_type &current_solution() const override;

    energy_type lower_bound() const override;

    bool is_partial_optimal(std::size_t variable_index) const override;

  private:
    void ensure_model_compatibility();
    void build_higher_order_energy(HocrBase *hocr);
    parameters_type parameters_;
    SolutionValue best_solution_value_;
    solution_type best_solution_;
    energy_type lower_bound_;
    std::vector<int> qpbo_labels_;
};

class HqpboDiscreteGmOptimizerFactory : public DiscreteGmOptimizerFactoryBase
{
  public:
    using factory_base_type = DiscreteGmOptimizerFactoryBase;
    virtual ~HqpboDiscreteGmOptimizerFactory() = default;
    std::unique_ptr<DiscreteGmOptimizerBase> create(const DiscreteGm &gm, OptimizerParameters &&params) const override
    {
        return std::make_unique<Hqpbo>(gm, std::move(params));
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
        return "Higher order QPBO optimizer";
    }
    OptimizerFlags flags() const override
    {
        return OptimizerFlags::PartialOptimal | OptimizerFlags::OptimalOnBinarySecondOrderSubmodular |
               OptimizerFlags::OptimalOnTrees;
    }
};

} // namespace nxtgm

XPLUGIN_CREATE_XPLUGIN_FACTORY(nxtgm::HqpboDiscreteGmOptimizerFactory);

namespace nxtgm
{
Hqpbo::Hqpbo(const DiscreteGm &gm, OptimizerParameters &&parameters)
    : base_type(gm, parameters),
      parameters_(parameters),
      best_solution_value_(),
      best_solution_(gm.num_variables(), 0),
      lower_bound_(std::numeric_limits<energy_type>::infinity()),
      qpbo_labels_(gm.num_variables(), -1)
{
    // check that there are no unknown parameters
    ensure_all_handled(name(), parameters);

    // check that the model is compatible (e.g. no constraints, binary labels, max order <= 10)
    ensure_model_compatibility();

    // make sure solution value is synced with solution
    best_solution_value_ = gm.evaluate(best_solution_, false /* early exit when infeasible*/);
}

void Hqpbo::build_higher_order_energy(HocrBase *hocr)
{

    const auto &gm = this->model();
    hocr->add_variables(gm.num_variables());

    std::vector<double> coeffs;
    coeffs.reserve(1 << max_arity);

    for (const auto &factor : gm.factors())
    {

        const unsigned int arity = factor.arity();
        if (arity == 1)
        {
            double values[2];
            factor.copy_energies(values);
            hocr->add_unary_term(values[1] - values[0], factor.variables()[0]);
        }
        else
        {
            auto num_assigments = 1 << arity; // 2^arity
            discrete_label_type clique_labels[arity];
            coeffs.resize(num_assigments);

            for (unsigned int subset = 1; subset < num_assigments; ++subset)
            {
                coeffs[subset] = 0;
            }

            for (unsigned int assignment = 0; assignment < num_assigments; ++assignment)
            {
                for (unsigned int i = 0; i < arity; ++i)
                {
                    if (assignment & (1 << i))
                    {
                        clique_labels[i] = 1;
                    }
                    else
                    {
                        clique_labels[i] = 0;
                    }
                }
                auto energy = factor(clique_labels);
                for (unsigned int subset = 1; subset < num_assigments; ++subset)
                {
                    if (assignment & ~subset)
                    {
                        continue;
                    }
                    else
                    {
                        int parity = 0;
                        for (unsigned int b = 0; b < arity; ++b)
                        {
                            parity ^= (((assignment ^ subset) & (1 << b)) != 0);
                        }
                        coeffs[subset] += parity ? -energy : energy;
                    }
                }
            }

            std::size_t vars[10];
            for (unsigned int subset = 1; subset < num_assigments; ++subset)
            {
                int degree = 0;
                for (unsigned int b = 0; b < arity; ++b)
                {
                    if (subset & (1 << b))
                    {
                        vars[degree++] = factor.variables()[b];
                    }
                }

                std::sort(vars, vars + degree);
                hocr->add_term(coeffs[subset], span<const std::size_t>(vars, degree));
            }
        }
    }
}

void Hqpbo::ensure_model_compatibility()
{
    const auto &gm = this->model();
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
    if (gm.max_arity() > max_arity)
    {
        std::stringstream ss;
        ss << "Hqpbo only supports factors with an arity <= " << max_arity << " (found " << gm.max_arity() << ")";
        throw std::runtime_error(ss.str());
    }
}

energy_type Hqpbo::lower_bound() const
{
    return lower_bound_;
}

OptimizationStatus Hqpbo::optimize_impl(reporter_callback_wrapper_type &reporter_callback,
                                        repair_callback_wrapper_type & /*repair_callback not used*/,
                                        const_discrete_solution_span starting_point)
{
    // shortcut to the model
    const auto &gm = this->model();

    // build higher order energy
    auto reducer_factory = get_plugin_registry<HocrFactoryBase>().get_factory(parameters_.hocr_plugin_name);
    auto reducer = reducer_factory->create();
    build_higher_order_energy(reducer.get());

    auto factory = get_plugin_registry<QpboFactoryBase>().get_factory(parameters_.qpbo_plugin_name);
    // this will only reserve variables but not add any
    auto qpbo = factory->create(gm.num_variables() * 2, 0);
    reducer->to_quadratic(qpbo.get());

    qpbo->merge_parallel_edges();
    qpbo->solve();

    if (!parameters_.strong_persistencies)
    {
        qpbo->compute_weak_persistencies();
    }
    lower_bound_ = qpbo->lower_bound();
    qpbo_labels_.resize(qpbo->num_nodes());
    qpbo->get_labels(qpbo_labels_.data());

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
    best_solution_value_ = gm.evaluate(best_solution_, false /* early exit when infeasible*/);

    return num_unlabeled == 0 ? OptimizationStatus::OPTIMAL : OptimizationStatus::CONVERGED;
}

SolutionValue Hqpbo::best_solution_value() const
{
    return best_solution_value_;
}
SolutionValue Hqpbo::current_solution_value() const
{
    return best_solution_value_;
}

const typename Hqpbo::solution_type &Hqpbo::best_solution() const
{
    return best_solution_;
}
const typename Hqpbo::solution_type &Hqpbo::current_solution() const
{
    return best_solution_;
}

bool Hqpbo::is_partial_optimal(std::size_t variable_index) const
{
    return qpbo_labels_[variable_index] >= 0;
}

} // namespace nxtgm
