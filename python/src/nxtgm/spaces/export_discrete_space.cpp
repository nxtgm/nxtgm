#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <nxtgm/spaces/discrete_space.hpp>

namespace py = pybind11;

namespace nxtgm {
void export_discrete_space(py::module_ &pymodule) {
  py::class_<DiscreteSpace>(pymodule, "DiscreteSpace")
      // operator[]
      .def("__getitem__", [](const DiscreteSpace &space,
                             std::size_t variable) { return space[variable]; })

      // size
      .def("__len__", [](const DiscreteSpace &space) { return space.size(); })

      // is_simple property (readonly)
      .def_property_readonly(
          "is_simple",
          [](const DiscreteSpace &space) { return space.is_simple(); })

      ;
}
} // namespace nxtgm
