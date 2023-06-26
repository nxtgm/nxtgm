#include <pybind11/pybind11.h>
#include <nxtgm/models/gm/discrete_gm.hpp>
#include <nxtgm/energy_functions/discrete_energy_functions.hpp>
#include <xtensor-python/pytensor.hpp>
#include <xtensor-python/pyarray.hpp>

namespace py = pybind11;

namespace nxtgm
{
    void export_discrete_gm(py::module_ & pymodule)
    {
        py::class_<DiscreteGm>(pymodule, "DiscreteGm")

            .def(py::init<std::size_t, discrete_label_type>(), py::arg("num_var"), py::arg("num_labels"))
            // general space constructor from numpy array

            // from xtensor python tensor
            .def(py::init([](const xt::pytensor<discrete_label_type, 1> & n_labels_array)
            {
                return new DiscreteGm(n_labels_array.begin(), n_labels_array.end());
            }), py::arg("n_labels"))

            
            // get space (reference with proper reference handling)
            .def_property_readonly("space", [](const DiscreteGm & gm)
            {
                return gm.space();
            }, py::return_value_policy::reference_internal)



            .def_property_readonly("max_factor_arity", [](const DiscreteGm & gm)
            {
                return gm.max_factor_arity();
            })
            .def_property_readonly("max_factor_size", [](const DiscreteGm & gm)
            {
                return gm.max_factor_size();
            })
            .def_property_readonly("max_constraint_arity", [](const DiscreteGm & gm)
            {
                return gm.max_constraint_arity();
            })
            .def_property_readonly("max_constraint_size", [](const DiscreteGm & gm)
            {
                return gm.max_constraint_size();
            })

            // evaluate from xt::pytensor
            .def("evaluate", [](const DiscreteGm & gm, const xt::pytensor<discrete_label_type, 1> & labels, bool early_exit_infeasible) {   
                span<const discrete_label_type> labels_span(labels.data(), labels.size());
                const SolutionValue value =  gm.evaluate(labels_span);
                return std::make_tuple(value.energy(), value.is_feasible(), value.how_violated());
            }, py::arg("labels"), py::arg("early_exit_infeasible") = false)


            .def("add_function", [](DiscreteGm & gm,  DiscreteEnergyFunctionBase * f)
            {
                auto cloned = f->clone();
                return gm.add_energy_function(std::move(cloned));
            }, py::arg("discrete_energy_function"))


            .def("add_function",[](DiscreteGm & gm, const xt::pytensor<energy_type, 1> & array)
            {
                auto f = std::make_unique<nxtgm::XTensor<1>>(array);
                return gm.add_energy_function(std::move(f));
            }, py::arg("energy_function"))

            .def("add_function",[](DiscreteGm & gm, const xt::pytensor<energy_type, 2> & array)
            {
                auto f = std::make_unique<nxtgm::XTensor<2>>(array);
                return gm.add_energy_function(std::move(f));
            }, py::arg("energy_function"))

            .def("add_function",[](DiscreteGm & gm, const xt::pyarray<energy_type> & array)
            {   
                auto f = std::make_unique<nxtgm::Xarray>(array);
                return gm.add_energy_function(std::move(f));
            }, py::arg("energy_function"))

            .def("add_factor", [](DiscreteGm & gm, const xt::pytensor<std::size_t, 1> & vis, std::size_t fid)
            {
                return gm.add_factor(vis, fid);
            }, py::arg("variables"), py::arg("function_id"))


        ;
    }
}