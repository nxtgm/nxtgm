#include <emscripten/bind.h>

#include <nxtgm/nxtgm.hpp>
#include <string>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

#include <nxtgm/models/gm/discrete_gm/gm.hpp>

#include <nxtgm/optimizers/gm/discrete/discrete_gm_optimizer_factory.hpp>
#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>

#include "convert.hpp"
#include "proposal_gen.hpp"

namespace nxtgm
{

namespace em = emscripten;

void export_optimizer()
{
    using solution_type = DiscreteGm::solution_type;

    em::class_<OptimizerParameters>("OptimizerParameters")
        .constructor<>()
        .function("set_string",
                  em::select_overload<void(OptimizerParameters &, const std::string &, const std::string &)>(
                      [](OptimizerParameters &self, const std::string &name, const std::string &value) {
                          self.string_parameters[name] = std::vector<std::string>(1, value);
                      }))
        .function("set_int", em::select_overload<void(OptimizerParameters &, const std::string &, int)>(
                                 [](OptimizerParameters &self, const std::string &name, int value) {
                                     self.int_parameters[name] = std::vector<int64_t>(1, value);
                                 }))
        .function("set_double", em::select_overload<void(OptimizerParameters &, const std::string &, double)>(
                                    [](OptimizerParameters &self, const std::string &name, double value) {
                                        self.double_parameters[name] = std::vector<double>(1, value);
                                    }))
        .function("set_parameter",
                  em::select_overload<void(OptimizerParameters &, const std::string &, const OptimizerParameters &)>(
                      [](OptimizerParameters &self, const std::string &name, const OptimizerParameters &value) {
                          self.optimizer_parameters[name] = std::vector<OptimizerParameters>(1, value);
                      }))

        .function("set", em::select_overload<void(OptimizerParameters &, const std::string &,
                                                  std::shared_ptr<ProposalGenFactoryBase>)>(
                             [](OptimizerParameters &self, const std::string &name,
                                std::shared_ptr<ProposalGenFactoryBase> value) { self.any_parameters[name] = value; }))

        .function("push_back_string",
                  em::select_overload<void(OptimizerParameters &, const std::string &, const std::string &)>(
                      [](OptimizerParameters &self, const std::string &name, const std::string &value) {
                          self.string_parameters[name].push_back(value);
                      }))
        .function("push_back_int", em::select_overload<void(OptimizerParameters &, const std::string &, int)>(
                                       [](OptimizerParameters &self, const std::string &name, int value) {
                                           self.int_parameters[name].push_back(value);
                                       }))
        .function("push_back_double", em::select_overload<void(OptimizerParameters &, const std::string &, double)>(
                                          [](OptimizerParameters &self, const std::string &name, double value) {
                                              self.double_parameters[name].push_back(value);
                                          }))
        .function("push_back_parameter",
                  em::select_overload<void(OptimizerParameters &, const std::string &, const OptimizerParameters &)>(
                      [](OptimizerParameters &self, const std::string &name, const OptimizerParameters &value) {
                          self.optimizer_parameters[name].push_back(value);
                      }))

        ;

    em::enum_<OptimizationStatus>("OptimizationStatus")
        .value("OPTIMAL", OptimizationStatus::OPTIMAL)
        .value("PARTIAL_OPTIMAL", OptimizationStatus::PARTIAL_OPTIMAL)
        .value("LOCAL_OPTIMAL", OptimizationStatus::LOCAL_OPTIMAL)
        .value("INFEASIBLE", OptimizationStatus::INFEASIBLE)
        .value("UNKNOWN", OptimizationStatus::UNKNOWN)
        .value("TIME_LIMIT_REACHED", OptimizationStatus::TIME_LIMIT_REACHED)
        .value("ITERATION_LIMIT_REACHED", OptimizationStatus::ITERATION_LIMIT_REACHED)
        .value("CONVERGED", OptimizationStatus::CONVERGED)
        .value("CALLBACK_EXIT", OptimizationStatus::CALLBACK_EXIT);

    using reporter_callback_base = ReporterCallbackBase<DiscreteGmOptimizerBase>;
    em::class_<DiscreteGmOptimizerBase>("DiscreteGmOptimizerBase")
        .function("optimize", em::select_overload<OptimizationStatus(DiscreteGmOptimizerBase &)>(
                                  [](DiscreteGmOptimizerBase &self) { return self.optimize(); }))

        .function("optimize",
                  em::select_overload<OptimizationStatus(DiscreteGmOptimizerBase &, reporter_callback_base *)>(
                      [](DiscreteGmOptimizerBase &self, reporter_callback_base *callback) {
                          return self.optimize(callback);
                      }),
                  em::allow_raw_pointers())

        // optimize with repoter and starting point
        .function("optimize",
                  em::select_overload<OptimizationStatus(DiscreteGmOptimizerBase &, reporter_callback_base *, em::val)>(
                      [](DiscreteGmOptimizerBase &self, reporter_callback_base *callback, em::val array) {
                          auto vector = vec_from_typed_array<uint32_t>(array);
                          solution_type starting_point = solution_type(vector.begin(), vector.end());
                          auto span = const_discrete_solution_span(starting_point.data(), starting_point.size());
                          return self.optimize(callback, nullptr, span);
                      }),
                  em::allow_raw_pointers())

        .function("best_solution",
                  em::select_overload<em::val(DiscreteGmOptimizerBase &)>([](DiscreteGmOptimizerBase &self) {
                      const auto &sol = self.best_solution();
                      return ptr_range_to_typed_array_copy(sol.data(), sol.size());
                  }))
        .function("current_solution",
                  em::select_overload<em::val(DiscreteGmOptimizerBase &)>([](DiscreteGmOptimizerBase &self) {
                      const auto &sol = self.best_solution();
                      return ptr_range_to_typed_array_copy(sol.data(), sol.size());
                  }))

        .function("best_solution_value", em::select_overload<SolutionValue(DiscreteGmOptimizerBase &)>(
                                             [](DiscreteGmOptimizerBase &self) { return self.best_solution_value(); }))
        .function("current_solution_value",
                  em::select_overload<SolutionValue(DiscreteGmOptimizerBase &)>(
                      [](DiscreteGmOptimizerBase &self) { return self.current_solution_value(); }));

    // em::function("discrete_gm_optimizer_factory",
    //              em::select_overload<std::unique_ptr<DiscreteGmOptimizerBase>(const DiscreteGm &,  std::string )>(
    //                  [](const DiscreteGm &gm,  std::string name ) {
    //                      const OptimizerParameters p;
    //                      auto expected = discrete_gm_optimizer_factory(gm, name, p);
    //                      if (!expected)
    //                      {
    //                          throw std::runtime_error(expected.error());
    //                      }
    //                      return std::move(expected.value());
    //                  }));
    em::function(
        "discrete_gm_optimizer_factory",
        em::select_overload<std::unique_ptr<DiscreteGmOptimizerBase>(DiscreteGm &, std::string, OptimizerParameters &)>(
            [](DiscreteGm &gm, std::string name, OptimizerParameters &p) -> std::unique_ptr<DiscreteGmOptimizerBase> {
                auto factory = get_discrete_gm_optimizer_factory(name);
                if (!factory)
                {
                    throw std::runtime_error("optimizer not found");
                }
                std::cout << "creating optimizer..." << std::endl;
                auto res = factory->create_unique(gm, OptimizerParameters(p));
                if (!res)
                {
                    std::cout << "error" << std::endl;
                    throw std::runtime_error("could not create optimizer");
                }
                return std::move(res);
            }));
}

} // namespace nxtgm
