#include <emscripten/bind.h>

#include <nxtgm/nxtgm.hpp>
#include <string>
#include <xtensor/xarray.hpp>

#include <nxtgm/constraint_functions/discrete_constraints.hpp>
#include <nxtgm/energy_functions/discrete_energy_functions.hpp>
#include <nxtgm/models/gm/discrete_gm/gm.hpp>
#include <nxtgm/spaces/discrete_space.hpp>

#include <nxtgm/optimizers/gm/discrete/discrete_gm_optimizer_factory.hpp>
#include <nxtgm/optimizers/gm/discrete/optimizer_base.hpp>

#include "callback.hpp"
#include "convert.hpp"

namespace nxtgm
{

namespace em = emscripten;

std::string get_exception_message(int exceptionPtr)
{
    return std::string(reinterpret_cast<std::exception *>(exceptionPtr)->what());
}

XArray *makeXArrayFromList(em::val value)
{
    require_array(value);
    const auto length = get_length(value);
    auto arr = xt::xarray<double>::from_shape({std::size_t(length)});
    for (int i = 0; i < length; i++)
    {
        arr(i) = as_number_checked<double>(value[i]);
    }
    return new XArray(std::move(arr));
}

// EM_JS(bool, is_arr, (em::val val), {
//     return (val instanceof Float64Array);
// });

XArray *makeXArrayShapeAndTypedArray(em::val shape, em::val typed_array)
{
    auto shape_vector = as_vector_of_numbers_checked<std::size_t>(shape);

    if (!em::val::module_property("_is_double_array")(typed_array).as<bool>())
    {
        throw std::runtime_error("typed_array must be Float64Array");
    }

    auto size = typed_array["length"].as<unsigned>();
    auto bytes_per_element = typed_array["BYTES_PER_ELEMENT"].as<unsigned>();
    auto xarr = xt::xarray<double>::from_shape(shape_vector);

    const unsigned byte_offset = typed_array["byteOffset"].as<em::val>().as<unsigned>();
    em::val js_array_buffer = typed_array["buffer"].as<em::val>();

    // this is a uint8 view of the array
    em::val js_uint8array = em::val::global("Uint8Array").new_(js_array_buffer, byte_offset, size * bytes_per_element);

    em::val wasm_heap_allocated =
        js_uint8array["constructor"].new_(em::val::module_property("HEAPU8")["buffer"],
                                          reinterpret_cast<uintptr_t>(xarr.data()), size * bytes_per_element);
    wasm_heap_allocated.call<void>("set", js_uint8array);

    return new XArray(std::move(xarr));
}

void export_space()
{
    em::class_<DiscreteSpace>("DiscreteSpace").constructor<std::size_t, std::size_t>();
}

void export_gm()
{
    em::class_<DiscreteGm>("DiscreteGm")

        .constructor<const DiscreteSpace &>()
        .constructor<std::size_t, std::size_t>()
        .function("save_binary", &DiscreteGm::save_binary)
        .class_function("load_binary", &DiscreteGm::load_binary)
        .function("as_json_str",
                  em::select_overload<std::string(DiscreteGm &, std::size_t)>([](DiscreteGm &self, std::size_t indent) {
                      const auto str = self.serialize_json().dump(indent);
                      return str;
                  }))

        .function("add_energy_function", em::select_overload<std::size_t(DiscreteGm &, DiscreteEnergyFunctionBase &)>(
                                             [](DiscreteGm &self, DiscreteEnergyFunctionBase &f) {
                                                 return self.add_energy_function(std::move(f.clone()));
                                             }))
        .function("add_factor", em::select_overload<std::size_t(DiscreteGm &, em::val, std::size_t)>(
                                    [](DiscreteGm &self, em::val variables, std::size_t fid) {
                                        // check that value is list
                                        if (!variables.isArray())
                                        {
                                            throw std::runtime_error("variables is not an array");
                                        }
                                        const auto length = variables["length"].as<int>();
                                        std::vector<std::size_t> vars(length);
                                        // check that value is list of numbers
                                        for (int i = 0; i < length; i++)
                                        {
                                            if (!variables[i].isNumber())
                                            {
                                                throw std::runtime_error("variables is not an array of numbers");
                                            }
                                            vars[i] = variables[i].as<std::size_t>();
                                        }

                                        return self.add_factor(vars, fid);
                                    }))

        // readonly properties
        .property("num_variables", &DiscreteGm::num_variables)
        .property("num_factors", &DiscreteGm::num_factors)
        .property("num_constraints", &DiscreteGm::num_constraints)

        .property("max_arity", &DiscreteGm::max_arity)
        .property("max_constraint_arity", &DiscreteGm::max_factor_arity)
        .property("max_constraint_arity", &DiscreteGm::max_constraint_arity)

        .property("max_constraint_size", &DiscreteGm::max_constraint_size)
        .property("max_factor_size", &DiscreteGm::max_factor_size);
}

void export_functions()
{
    em::class_<DiscreteEnergyFunctionBase>("DiscreteEnergyFunctionBase")

        .function("as_json_str", em::select_overload<std::string(DiscreteEnergyFunctionBase &, std::size_t)>(
                                     [](DiscreteEnergyFunctionBase &self, std::size_t indent) {
                                         const auto str = self.serialize_json().dump(indent);
                                         return str;
                                     }));
    em::class_<DiscreteConstraintFunctionBase>("DiscreteConstraintFunctionBase");

    em::class_<Potts, em::base<DiscreteEnergyFunctionBase>>("Potts").constructor<std::size_t, energy_type>();

    em::class_<XArray, em::base<DiscreteEnergyFunctionBase>>("XArray")
        .constructor(&makeXArrayFromList, em::allow_raw_pointers())
        .constructor(&makeXArrayShapeAndTypedArray, em::allow_raw_pointers());
}

void export_optimizer()
{
    using solution_type = DiscreteGm::solution_type;

    em::class_<OptimizerParameters>("OptimizerParameters")
        .constructor<>()
        .function("set_string",
                  em::select_overload<void(OptimizerParameters &, const std::string &, const std::string &)>(
                      [](OptimizerParameters &self, const std::string &name, const std::string &value) {
                          self.string_parameters[name] = value;
                      }))
        .function("set_int", em::select_overload<void(OptimizerParameters &, const std::string &, int)>(
                                 [](OptimizerParameters &self, const std::string &name, int value) {
                                     self.int_parameters[name] = value;
                                 }))
        .function("set_double", em::select_overload<void(OptimizerParameters &, const std::string &, double)>(
                                    [](OptimizerParameters &self, const std::string &name, double value) {
                                        self.double_parameters[name] = value;
                                    }))
        .function("set_parameter",
                  em::select_overload<void(OptimizerParameters &, const std::string &, const OptimizerParameters &)>(
                      [](OptimizerParameters &self, const std::string &name, const OptimizerParameters &value) {
                          self.optimizer_parameters[name] = value;
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

    em::class_<DiscreteGmOptimizerBase>("DiscreteGmOptimizerBase")
        .function("optimize", em::select_overload<OptimizationStatus(DiscreteGmOptimizerBase &)>(
                                  [](DiscreteGmOptimizerBase &self) { return self.optimize(); }))

        .function("best_solution",
                  em::select_overload<em::val(DiscreteGmOptimizerBase &)>([](DiscreteGmOptimizerBase &self) {
                      const auto &sol = self.best_solution();
                      return ptr_range_to_typed_array(sol.data(), sol.size());
                  }));

    em::function(
        "discrete_gm_optimizer_factory",
        em::select_overload<std::unique_ptr<DiscreteGmOptimizerBase>(
            const DiscreteGm &, const std::string &, const OptimizerParameters &)>(&discrete_gm_optimizer_factory));
}

EMSCRIPTEN_BINDINGS(my_module)
{

    em::register_vector<discrete_label_type>("VectorUInt16");
    em::function("get_exception_message", &get_exception_message);

    export_space();
    export_gm();
    export_functions();
    export_optimizer();
    export_callbacks();
}

} // namespace nxtgm
