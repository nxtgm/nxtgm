#include <nxtgm/optimizers/callbacks.hpp>
#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>
#include <pybind11/chrono.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <xtensor-python/pytensor.hpp>

#include <nxtgm/optimizers/gm/discrete/discrete_gm_optimizer_factory.hpp>

// plugin factories need to be stored in any
#include <nxtgm/plugins/proposal_gen/proposal_gen_base.hpp>

namespace py = pybind11;

namespace nxtgm
{

class PyReporterCallbackBase : public ReporterCallbackBase<DiscreteGmOptimizerBase>
{
  public:
    using base_type = ReporterCallbackBase<DiscreteGmOptimizerBase>;
    /* Inherit the constructors */
    using base_type::base_type;

    /* Trampoline (need one for each virtual function) */
    void begin() override
    {
        PYBIND11_OVERRIDE_PURE(void,      /* Return type */
                               base_type, /* Parent class */
                               begin,     /* Name of function in C++ (must match Python name) */
        );
    }
    void end() override
    {
        PYBIND11_OVERRIDE_PURE(void,      /* Return type */
                               base_type, /* Parent class */
                               end,       /* Name of function in C++ (must match Python name) */
        );
    }
    bool report() override
    {
        PYBIND11_OVERRIDE_PURE(bool,      /* Return type */
                               base_type, /* Parent class */
                               report,    /* Name of function in C++ (must match Python name) */
        );
    }
    bool report_data(const ReportData &data) override
    {
        PYBIND11_OVERRIDE_PURE(bool,        /* Return type */
                               base_type,   /* Parent class */
                               report_data, /* Name of function in C++ (must match Python name) */
                               data);
    }
};

void export_discrete_gm_optimizers(py::module_ &pymodule)
{

    py::enum_<OptimizationStatus>(pymodule, "OptimizationStatus")
        .value("OPTIMAL", OptimizationStatus::OPTIMAL)
        .value("PARTIAL_OPTIMAL", OptimizationStatus::PARTIAL_OPTIMAL)
        .value("LOCAL_OPTIMAL", OptimizationStatus::LOCAL_OPTIMAL)
        .value("INFEASIBLE", OptimizationStatus::INFEASIBLE)
        .value("UNKNOWN", OptimizationStatus::UNKNOWN)
        .value("TIME_LIMIT_REACHED", OptimizationStatus::TIME_LIMIT_REACHED)
        .value("CALLBACK_EXIT", OptimizationStatus::CALLBACK_EXIT);

    // we will create a class OptimizerParameters derived from _OptimizerParameters
    // on the python side to allow for a more pythonic interface
    py::class_<OptimizerParameters>(pymodule, "_OptimizerParameters")
        .def(py::init<>())
        .def("__setitem__",
             [](OptimizerParameters &params, const std::string &key, const std::string &value) {
                 params.string_parameters[key] = std::vector<std::string>(1, value);
             })
        .def("__setitem__", [](OptimizerParameters &params, const std::string &key,
                               int value) { params.int_parameters[key] = std::vector<int64_t>(1, value); })
        .def("__setitem__", [](OptimizerParameters &params, const std::string &key,
                               double value) { params.double_parameters[key] = std::vector<double>(1, value); })
        .def("__setitem__",
             [](OptimizerParameters &params, const std::string &key, const OptimizerParameters &value) {
                 params.optimizer_parameters[key] = std::vector<OptimizerParameters>(1, value);
             })
        .def("__setitem__",
             [](OptimizerParameters &params, const std::string &key, std::shared_ptr<ProposalGenFactoryBase> value) {
                 params.any_parameters[key] = std::shared_ptr<ProposalGenFactoryBase>(value);
             });

    py::class_<ReportData>(pymodule, "ReportData")
        .def(py::init<>())
        .def("get_double",
             [](const ReportData &data, const std::string &key) {
                 auto it = data.double_data.find(key);
                 if (it == data.double_data.end())
                 {
                     throw std::runtime_error("key not found");
                 }
                 return py::array(it->second.size(), it->second.data(),
                                  py::cast(data, py::return_value_policy::reference));
             })
        .def("get_int", [](const ReportData &data, const std::string &key) {
            auto it = data.int_data.find(key);
            if (it == data.int_data.end())
            {
                throw std::runtime_error("key not found");
            }
            return py::array(it->second.size(), it->second.data(), py::cast(data, py::return_value_policy::reference));
        });

    using reporter_callback_base_type = DiscreteGmOptimizerBase::reporter_callback_base_type;
    py::class_<reporter_callback_base_type, PyReporterCallbackBase /* <--- trampoline*/>(
        pymodule, "DiscreteGmOptimizerReporterCallbackBase")
        .def(py::init<const DiscreteGmOptimizerBase *>(), py::arg("optimizer"), py::keep_alive<0, 1>())
        .def("begin", &reporter_callback_base_type::begin)
        .def("end", &reporter_callback_base_type::end)
        .def("report", &reporter_callback_base_type::report)
        .def("report_data", &reporter_callback_base_type::report_data);

    using reporter_callback = ReporterCallback<DiscreteGmOptimizerBase>;
    py::class_<reporter_callback, reporter_callback_base_type>(pymodule, "DiscreteGmOptimizerReporterCallback")
        .def(py::init<const DiscreteGmOptimizerBase *>(), py::arg("optimizer"), py::keep_alive<0, 1>());

    // optimizer base
    py::class_<DiscreteGmOptimizerBase>(pymodule, "DiscreteGmOptimizerBase")

        .def(
            "optimize",
            [](DiscreteGmOptimizerBase *optimizer, reporter_callback_base_type *reporter_callback,
               xt::pytensor<discrete_label_type, 1> stating_point) {
                if (stating_point.size() != 0)
                {
                    auto ptr = stating_point.data();
                    span<const discrete_label_type> span(ptr, ptr + stating_point.size());
                    return optimizer->optimize(reporter_callback, nullptr, span);
                }
                else
                {
                    return optimizer->optimize(reporter_callback);
                }
            },
            py::arg("reporter_callback") = nullptr, py::arg("starting_point") = xt::pytensor<discrete_label_type, 1>())
        .def("best_solution",
             [](DiscreteGmOptimizerBase *optimizer) {
                 const auto &sol = optimizer->best_solution();
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

    pymodule.def(
        "_discrete_gm_optimizer_factory",
        [](const DiscreteGm &gm, const std::string &optimizer_name, const OptimizerParameters &parameters) {
            auto expected = discrete_gm_optimizer_factory(gm, optimizer_name, parameters);
            if (!expected)
            {

                throw std::runtime_error(expected.error());
            }
            return std::move(expected.value());
        },
        py::arg("gm"), py::arg("optimizer_name"), py::arg("parameters") = OptimizerParameters(),
        py::keep_alive<1, 2>());
}
} // namespace nxtgm
