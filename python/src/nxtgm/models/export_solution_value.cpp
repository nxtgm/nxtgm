#include <nxtgm/models/solution_value.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace nxtgm
{
void export_solution_value(py::module_ &pymodule)
{
    py::class_<SolutionValue>(pymodule, "SolutionValue")
        // readonly properties

        .def_property_readonly("energy", [](const SolutionValue &solution) { return solution.energy(); })
        .def_property_readonly("is_feasible", &SolutionValue::is_feasible)
        .def_property_readonly("how_violated", [](const SolutionValue &solution) { return solution.how_violated(); })
        // operator <
        .def("__lt__", [](const SolutionValue &solution, const SolutionValue &other) { return solution < other; })
        .def("__gt__", [](const SolutionValue &solution, const SolutionValue &other) { return other < solution; })
        .def("__leq__", [](const SolutionValue &solution, const SolutionValue &other) { return solution <= other; })
        .def("__geq__", [](const SolutionValue &solution, const SolutionValue &other) { return !(solution < other); })

        // as tuple
        .def("__tuple__",
             [](const SolutionValue &solution) { return py::make_tuple(solution.energy(), solution.how_violated()); })
        .def("__len__", [](const SolutionValue &solution) { return 2; })
        // getitme
        .def("__getitem__",
             [](const SolutionValue &solution, std::size_t index) -> py::object {
                 switch (index)
                 {
                 case 0:
                     // as py::object
                     return py::cast(solution.energy());
                 case 1:
                     return py::cast(solution.how_violated());
                 default:
                     throw std::out_of_range("index out of range");
                 }
             })

        // print / str
        .def("__str__",
             [](const SolutionValue &solution) {
                 std::stringstream ss;
                 ss << "(energy=" << solution.energy() << ", how_violated=" << solution.how_violated() << ")";
                 return ss.str();
             })

        ;
}
} // namespace nxtgm
