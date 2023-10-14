#include <nxtgm/functions/discrete_constraint_function_base.hpp>
#include <nxtgm/functions/discrete_constraints.hpp>
#include <pybind11/pybind11.h>
#include <xtensor-python/pyarray.hpp>
#include <xtensor-python/pytensor.hpp>

namespace py = pybind11;

namespace nxtgm
{
void export_discrete_constraint_functions(py::module_ &pymodule)
{

    py::class_<DiscreteConstraintFunctionBase>(pymodule, "DiscreteConstraintFunctionBase")

        // read-only properties arity
        .def_property_readonly("arity", &DiscreteConstraintFunctionBase::arity)
        .def_property_readonly("size", &DiscreteConstraintFunctionBase::size)
        .def("shape", &DiscreteConstraintFunctionBase::shape)

        // return shape as tuple
        .def_property_readonly("shape",
                               [](const DiscreteConstraintFunctionBase *self) {
                                   const auto arity = self->arity();
                                   auto my_tuple = py::tuple(arity);

                                   for (std::size_t i = 0; i < arity; ++i)
                                   {
                                       my_tuple[i] = self->shape(i);
                                   }
                                   return my_tuple;
                               })

        .def("__getitem__",
             [](DiscreteConstraintFunctionBase *self, uint16_t label) {
                 const auto arity = self->arity();
                 if (arity > 1)
                 {
                     throw std::runtime_error("only unary energy functions can be "
                                              "accessed with a single index");
                 }
                 return self->value(&label);
             })

        .def("__getitem__",
             [](DiscreteConstraintFunctionBase *self, py::tuple indices) {
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
                 return self->value(index.data());
             })

        ;

    py::class_<UniqueLables, DiscreteConstraintFunctionBase>(pymodule, "UniqueLables")
        .def(py::init<std::size_t, discrete_label_type, energy_type>(), py::arg("arity"), py::arg("num_labels"),
             py::arg("scale") = 1.0);

    py::class_<ArrayDiscreteConstraintFunction, DiscreteConstraintFunctionBase>(pymodule,
                                                                                "ArrayDiscreteConstraintFunction")
        // custom init from numpy via xt::pyarray from labmda
        .def(py::init([](xt::pyarray<energy_type> a) { return new ArrayDiscreteConstraintFunction(a); }));
}

} // namespace nxtgm
