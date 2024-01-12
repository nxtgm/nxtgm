#include <memory>
#include <nxtgm/functions/label_count_constraint_base.hpp>
#include <nxtgm/nxtgm.hpp>
#include <nxtgm/optimizers/gm/discrete/discrete_gm_optimizer_factory.hpp>
#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>
#include <nxtgm/plugins/ilp/ilp_base.hpp>
#include <nxtgm/plugins/proposal_gen/proposal_gen_base.hpp>

#include <string>

// xplugin
#include <xplugin/xfactory.hpp>

namespace nxtgm
{

class ProposalGenOptimizerBased : public ProposalGenBase
{
    class Callback : public ReporterCallbackBase<DiscreteGmOptimizerBase>
    {
      public:
        virtual ~Callback() = default;

        Callback(DiscreteGmOptimizerBase *optimizer, discrete_label_type *proposal, const discrete_label_type *best,
                 std::function<ProposalConsumerStatus()> consumer)
            : ReporterCallbackBase<DiscreteGmOptimizerBase>(optimizer),
              proposal_(proposal),
              best_(best),
              consumer_(consumer)
        {
            const auto gm = optimizer->model();
            // find global matching constraints
            for (const auto &constraint : gm.constraints())
            {
                if (constraint.arity() == gm.num_variables())
                {
                    const LabelCountConstraintBase *label_count_constraint =
                        dynamic_cast<const LabelCountConstraintBase *>(constraint.function());
                    if (label_count_constraint)
                    {
                        constraint_function_ = label_count_constraint;
                        break;
                    }
                }
            }
            if (!constraint_function_)
            {
                throw std::runtime_error("no global matching constraint found");
            }
            indicator_vartiable_mapping_.resize(gm.num_variables());
            for (std::size_t vi = 0; vi < gm.num_variables(); ++vi)
            {
                indicator_vartiable_mapping_[vi] = vi * gm.space().max_num_labels();
            }
        }

        void begin() override
        {
            //
        }
        void end() override
        {
            // consumer_();
        }
        bool report() override
        {
            throw std::runtime_error("report() should not be called");
        }

        bool report_data(const ReportData &data) override
        {
            // get beliefs
            auto beliefs = data.double_data.find("beliefs")->second;
            const auto beliefs_size = beliefs.size();
            const auto belief_ptr = beliefs.data();

            const auto num_var = this->optimizer()->model().num_variables();
            const auto num_labels = this->optimizer()->model().space().max_num_labels();

            if (beliefs_size != num_var * num_labels)
            {
                throw std::runtime_error("beliefs size mismatch");
            }

            // apply unique label constraints
            IlpData ilp_data;
            ilp_data.add_variables(num_var * num_labels,
                                   /*lower bound*/ 0,
                                   /*upper bound*/ 1,
                                   /*objective*/ 0.0,
                                   /*is_integer*/ false);

            // add  marginalization constraints and objective
            std::size_t ilp_var = 0;
            for (std::size_t vi = 0; vi < num_var; ++vi)
            {
                ilp_data.begin_constraint(1, 1);
                for (std::size_t label = 0; label < num_labels; ++label)
                {
                    ilp_data.add_constraint_coefficient(vi * num_labels + label, 1);
                    ilp_data[ilp_var] = belief_ptr[ilp_var];
                    ++ilp_var;
                }
            }

            // add global matching constraints
            constraint_function_->add_to_lp(ilp_data, indicator_vartiable_mapping_.data());

            // solve ILP
            OptimizerParameters parameters;
            parameters["integer"] = false;
            auto factory = get_plugin_registry<IlpFactoryBase>().get_factory(std::string("ilp_") + "highs");
            auto ilp_solver = factory->create(std::move(ilp_data), std::move(parameters));
            ilp_solver->optimize(nullptr);
            std::vector<double> solution(num_var * num_labels);
            ilp_solver->get_solution(solution.data());

            // set proposal
            for (std::size_t vi = 0; vi < num_var; ++vi)
            {
                for (std::size_t label = 0; label < num_labels; ++label)
                {
                    const auto lp_sol = solution[vi * num_labels + label];
                    if (lp_sol >= 0.9)
                    {
                        proposal_[vi] = label;
                        break;
                    }
                }
            }

            auto res = consumer_();
            if (res == ProposalConsumerStatus::ACCEPTED)
            {
                // std::cout<<"accepted"<<std::endl;
            }
            else if (res == ProposalConsumerStatus::REJECTED)
            {
                // std::cout<<"rejected"<<std::endl;
            }
            else if (res == ProposalConsumerStatus::EXIT)
            {
                // std::cout<<"exit"<<std::endl;
                return false;
            }
            return true;
        }

      private:
        discrete_label_type *proposal_;
        const discrete_label_type *best_;
        std::function<ProposalConsumerStatus()> consumer_;

        const LabelCountConstraintBase *constraint_function_ = nullptr;
        std::vector<std::size_t> indicator_vartiable_mapping_;
    };

    class parameters_type
    {
      public:
        inline parameters_type(OptimizerParameters &parameters)
        {
            parameters.assign_and_pop("belief_propagation_parameters", belief_propagation_parameters);
        }

        OptimizerParameters belief_propagation_parameters;
    };

  public:
    ~ProposalGenOptimizerBased() = default;
    ProposalGenOptimizerBased(const DiscreteGm &gm, OptimizerParameters &&parameters)
        : ProposalGenBase(),
          gm_(gm),
          parameters_(parameters)
    {
        ensure_all_handled("ProposalGenOptimizerBased", parameters);
    }

    void generate(const discrete_label_type *best, discrete_label_type *proposal,
                  std::function<ProposalConsumerStatus()> consumer) override
    {

        auto expected_optimizer = discrete_gm_optimizer_factory(gm_, "belief_propagation",
                                                                std::move(parameters_.belief_propagation_parameters));
        if (!expected_optimizer)
        {
            throw std::runtime_error(expected_optimizer.error());
        }
        auto optimizer = std::move(expected_optimizer.value());
        Callback callback(optimizer.get(), proposal, best, consumer);

        const_discrete_solution_span starting_point(proposal, gm_.num_variables());

        optimizer->optimize(&callback, nullptr, starting_point);
    }

  private:
    const DiscreteGm &gm_;
    parameters_type parameters_;
    discrete_label_type max_num_labels_;
    discrete_label_type current_alpha_;
};

class ProposalGenOptimizerBasedFactory : public ProposalGenFactoryBase
{
  public:
    using factory_base_type = ProposalGenFactoryBase;
    ProposalGenOptimizerBasedFactory() = default;
    ~ProposalGenOptimizerBasedFactory() = default;

    std::unique_ptr<ProposalGenBase> create(const DiscreteGm &gm, OptimizerParameters &&parameters) override
    {
        return std::make_unique<ProposalGenOptimizerBased>(gm, std::move(parameters));
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
        return "Matching Bp";
    }
};

} // namespace nxtgm

XPLUGIN_CREATE_XPLUGIN_FACTORY(nxtgm::ProposalGenOptimizerBasedFactory);
