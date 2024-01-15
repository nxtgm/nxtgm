#include <nxtgm/functions/array_constraint_function.hpp>
#include <nxtgm/functions/discrete_energy_function_base.hpp>
#include <nxtgm/functions/xarray_energy_function.hpp>
#include <nxtgm/functions/xtensor_energy_function.hpp>
#include <nxtgm/models/gm/discrete_gm.hpp>
#include <pybind11/pybind11.h>
#include <xtensor-python/pyarray.hpp>
#include <xtensor-python/pytensor.hpp>

namespace py = pybind11;

namespace nxtgm
{
void export_discrete_gm(py::module_ &pymodule)
{

    using DiscreteGmConstRefWrapper = std::reference_wrapper<const DiscreteGm>;

    py::class_<DiscreteGm>(pymodule, "DiscreteGm")

        .def(py::init<std::size_t, discrete_label_type>(), py::arg("num_var"), py::arg("num_labels"))
        // general space constructor from numpy array

        // from xtensor python tensor
        .def(py::init([](const xt::pytensor<discrete_label_type, 1> &n_labels_array) {
                 return new DiscreteGm(n_labels_array.begin(), n_labels_array.end());
             }),
             py::arg("n_labels"))

        // get space (reference with proper reference handling)
        .def_property_readonly(
            "space", [](const DiscreteGm &gm) { return gm.space(); }, py::return_value_policy::reference_internal)

        .def_property_readonly("num_variables", [](const DiscreteGm &gm) { return gm.num_variables(); })
        .def_property_readonly("num_factors", [](const DiscreteGm &gm) { return gm.num_factors(); })
        .def_property_readonly("num_constraints", [](const DiscreteGm &gm) { return gm.num_constraints(); })
        .def_property_readonly("max_factor_arity", [](const DiscreteGm &gm) { return gm.max_factor_arity(); })
        .def_property_readonly("max_factor_size", [](const DiscreteGm &gm) { return gm.max_factor_size(); })
        .def_property_readonly("max_constraint_arity", [](const DiscreteGm &gm) { return gm.max_constraint_arity(); })
        .def_property_readonly("max_constraint_size", [](const DiscreteGm &gm) { return gm.max_constraint_size(); })

        // evaluate from xt::pytensor
        .def(
            "evaluate",
            [](const DiscreteGm &gm, const xt::pytensor<discrete_label_type, 1> &labels, bool early_exit_infeasible) {
                span<const discrete_label_type> labels_span(labels.data(), labels.size());
                return gm.evaluate(labels_span);
            },
            py::arg("labels"), py::arg("early_exit_infeasible") = false)

        .def(
            "add_function",
            [](DiscreteGm &gm, DiscreteEnergyFunctionBase *f) {
                auto cloned = f->clone();
                return gm.add_energy_function(std::move(cloned));
            },
            py::arg("discrete_energy_function"))

        // .def(
        //     "add_function",
        //     [](DiscreteGm &gm, const xt::pytensor<energy_type, 1> &array) {
        //         auto f = std::make_unique<nxtgm::XTensor<1>>(array);
        //         return gm.add_energy_function(std::move(f));
        //     },
        //     py::arg("energy_function"))

        // .def(
        //     "add_function",
        //     [](DiscreteGm &gm, const xt::pytensor<energy_type, 2> &array) {
        //         auto f = std::make_unique<nxtgm::XTensor<2>>(array);
        //         return gm.add_energy_function(std::move(f));
        //     },
        //     py::arg("energy_function"))

        .def(
            "add_function",
            [](DiscreteGm &gm, const xt::pyarray<energy_type> &array) {
                auto f = std::make_unique<nxtgm::XArray>(array);
                return gm.add_energy_function(std::move(f));
            },
            py::arg("energy_function"))

        .def(
            "add_factor",
            [](DiscreteGm &gm, const xt::pytensor<std::size_t, 1> &vis, std::size_t fid) {
                return gm.add_factor(vis, fid);
            },
            py::arg("variables"), py::arg("function_id"))

        .def(
            "add_constraint_function",
            [](DiscreteGm &gm, DiscreteConstraintFunctionBase *f) {
                auto cloned = f->clone();
                return gm.add_constraint_function(std::move(cloned));
            },
            py::arg("discrete_constraint_function"))

        .def(
            "add_contraint_function",
            [](DiscreteGm &gm, const xt::pyarray<energy_type> &array) {
                auto f = std::make_unique<nxtgm::ArrayDiscreteConstraintFunction>(array);
                return gm.add_constraint_function(std::move(f));
            },
            py::arg("discrete_constraint_function"))

        .def(
            "add_constraint",
            [](DiscreteGm &gm, const xt::pytensor<std::size_t, 1> &vis, std::size_t fid) {
                return gm.add_constraint(vis, fid);
            },
            py::arg("variables"), py::arg("function_id"))

        ;
}
} // namespace nxtgm
