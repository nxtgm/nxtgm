#define FORCE_IMPORT_ARRAY

#include <xtensor-python/pytensor.hpp>


#include <pybind11/pybind11.h>
#include <nxtgm/version.hpp>
#include <nxtgm/nxtgm.hpp>


namespace py = pybind11;

namespace nxtgm
{   
    void export_discrete_energy_functions(py::module_ & pymodule);
    void export_discrete_space(py::module_ & pymodule);
    void export_discrete_gm(py::module_ & pymodule);

    void export_nxtgm(py::module_ pymodule)
    {   
        export_discrete_energy_functions(pymodule);
        export_discrete_space(pymodule);
        export_discrete_gm(pymodule);
    }
}




PYBIND11_MODULE(_nxtgm, m) {
    m.doc() = R"pbdoc(
        _nxtgm core module
        -----------------------

        .. currentmodule:: _nxgtgm

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";


    m.attr("__version__") = NXTGM_VERSION;
    xt::import_numpy();
    nxtgm::export_nxtgm(m);
}