#include <nxtgm/optimizers/callbacks.hpp>
#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>
#include <pybind11/chrono.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11_json/pybind11_json.hpp>

#include <xtensor-python/pytensor.hpp>

#include <nxtgm/optimizers/gm/discrete/discrete_gm_optimizer_factory.hpp>

namespace py = pybind11;

namespace nxtgm
{
template <class optimzier_type>
auto export_optimizer(py::module_ &pymodule)
{
    auto optimizer_cls = py::class_<optimzier_type, DiscreteGmOptimizerBase>(pymodule, optimzier_type::name().c_str(),
                                                                             py::dynamic_attr())
                             .def(py::init<const DiscreteGm &, const nlohmann::json &>(), py::arg("gm"),
                                  py::arg("parameters") = nlohmann::json(), py::keep_alive<0, 1>());
}

void export_discrete_gm_optimizers(py::module_ &pymodule)
{
    // enum
    py::enum_<OptimizationStatus>(pymodule, "OptimizationStatus")
        .value("OPTIMAL", OptimizationStatus::OPTIMAL)
        .value("PARTIAL_OPTIMAL", OptimizationStatus::PARTIAL_OPTIMAL)
        .value("LOCAL_OPTIMAL", OptimizationStatus::LOCAL_OPTIMAL)
        .value("INFEASIBLE", OptimizationStatus::INFEASIBLE)
        .value("UNKNOWN", OptimizationStatus::UNKNOWN)
        .value("TIME_LIMIT_REACHED", OptimizationStatus::TIME_LIMIT_REACHED)
        .value("CALLBACK_EXIT", OptimizationStatus::CALLBACK_EXIT);

    // reporter callback base
    using reporter_callback_base_type = DiscreteGmOptimizerBase::reporter_callback_base_type;
    py::class_<reporter_callback_base_type>(pymodule, "DiscreteGmOptimizerReporterCcallbackBase");

    using reporter_callback = ReporterCallback<DiscreteGmOptimizerBase>;
    py::class_<reporter_callback, reporter_callback_base_type>(pymodule, "DiscreteGmOptimizerReporterCallback")
        .def(py::init<const DiscreteGmOptimizerBase *>(), py::arg("optimizer"), py::keep_alive<0, 1>());

    // optimizer base
    py::class_<DiscreteGmOptimizerBase>(pymodule, "DiscreteGmOptimizerBase")

        .def("optimize", [](DiscreteGmOptimizerBase *optimizer) { return optimizer->optimize(); })
        .def("optimize",
             [](DiscreteGmOptimizerBase *optimizer, reporter_callback_base_type *reporter_callback) {
                 return optimizer->optimize(reporter_callback);
             })
        .def("best_solution",
             [](DiscreteGmOptimizerBase *optimizer) {
                 const auto &sol = optimizer->best_solution();
                 std::cout << sol.size() << std::endl;
                 xt::pytensor<discrete_label_type, 1> array =
                     xt::zeros<discrete_label_type>({uint16_t(optimizer->model().num_variables())});
                 std::copy(sol.begin(), sol.end(), array.begin());
                 return array;
             })
        .def("current_solution",
             [](DiscreteGmOptimizerBase *optimizer) {
                 const auto &sol = optimizer->current_solution();
                 xt::pytensor<discrete_label_type, 1> array =
                     xt::zeros<discrete_label_type>({uint16_t(optimizer->model().num_variables())});
                 std::copy(sol.begin(), sol.end(), array.begin());
                 return array;
             })
        .def("best_solution_value", &DiscreteGmOptimizerBase::best_solution_value)
        .def("current_solution_value", &DiscreteGmOptimizerBase::current_solution_value)
        .def("lower_bound", &DiscreteGmOptimizerBase::lower_bound);

    pymodule.def("discrete_gm_optimizer_factory", &discrete_gm_optimizer_factory, py::arg("optimizer_name"),
                 py::arg("gm"), py::arg("parameters") = nlohmann::json(), py::keep_alive<1, 2>());

    // concrete optimizers
    // export_optimizer<BruteForceNaive>(pymodule);
    // export_optimizer<IlpHighs>(pymodule);
    // export_optimizer<BeliefPropagation>(pymodule);
    // export_optimizer<DynamicProgramming>(pymodule);
    // export_optimizer<Icm>(pymodule);
    // export_optimizer<MatchingIcm>(pymodule);
}
} // namespace nxtgm
