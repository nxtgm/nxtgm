#include <nxtgm/energy_functions/discrete_energy_function_base.hpp>
#include <nxtgm/energy_functions/discrete_energy_functions.hpp>
#include <pybind11/pybind11.h>
#include <xtensor-python/pytensor.hpp>

namespace py = pybind11;

namespace nxtgm
{
void export_discrete_energy_functions(py::module_& pymodule)
{

    py::class_<DiscreteEnergyFunctionBase>(pymodule,
                                           "DiscreteEnergyFunctionBase")

        // read-only properties arity
        .def_property_readonly("arity", &DiscreteEnergyFunctionBase::arity)
        .def_property_readonly("size", &DiscreteEnergyFunctionBase::size)
        .def("shape", &DiscreteEnergyFunctionBase::shape)

        // return shape as tuple
        .def_property_readonly("shape",
                               [](const DiscreteEnergyFunctionBase* self)
                               {
                                   const auto arity = self->arity();
                                   auto my_tuple = py::tuple(arity);

                                   for (std::size_t i = 0; i < arity; ++i)
                                   {
                                       my_tuple[i] = self->shape(i);
                                   }
                                   return my_tuple;
                               })

        .def("__getitem__",
             [](DiscreteEnergyFunctionBase* self, uint16_t label)
             {
                 const auto arity = self->arity();
                 if (arity > 1)
                 {
                     throw std::runtime_error(
                         "only unary energy functions can be "
                         "accessed with a single index");
                 }
                 return self->energy(&label);
             })

        .def("__getitem__",
             [](DiscreteEnergyFunctionBase* self, py::tuple indices)
             {
                 const auto arity = self->arity();
                 // check that the tuple has the correct arity
                 if (indices.size() != arity)
                 {
                     throw std::runtime_error("tuple must have the same arity "
                                              "as the energy function");
                 }

                 // create the index vector
                 small_arity_vector<discrete_label_type> index(arity);
                 for (std::size_t i = 0; i < arity; ++i)
                 {
                     index[i] = indices[i].cast<discrete_label_type>();
                 }

                 // return the value
                 return self->energy(index.data());
             })

        ;

    py::class_<Potts, DiscreteEnergyFunctionBase>(pymodule, "Potts")
        .def(py::init<std::size_t, energy_type>(), py::arg("num_labels"),
             py::arg("beta"));

    py::class_<LabelCosts, DiscreteEnergyFunctionBase>(pymodule, "LabelCosts")
        .def(py::init(
                 [](std::size_t arity,
                    const xt::pytensor<energy_type, 1>& values) {
                     return new LabelCosts(arity, values.begin(), values.end());
                 }),
             py::arg("arity"), py::arg("label_costs"))

        ;
}
} // namespace nxtgm
