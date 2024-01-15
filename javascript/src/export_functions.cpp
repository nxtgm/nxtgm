#include <emscripten/bind.h>

#include <nxtgm/nxtgm.hpp>
#include <string>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

#include <nxtgm/functions/array_constraint_function.hpp>
#include <nxtgm/functions/potts_energy_function.hpp>
#include <nxtgm/functions/unique_labels_constraint_function.hpp>
#include <nxtgm/functions/xarray_energy_function.hpp>
#include <nxtgm/functions/xtensor_energy_function.hpp>

#include <nxtgm/models/gm/discrete_gm/gm.hpp>

#include "convert.hpp"

namespace nxtgm
{

namespace em = emscripten;

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

void export_functions()
{
    em::class_<DiscreteEnergyFunctionBase>("DiscreteEnergyFunctionBase")

        .function("as_json_str", em::select_overload<std::string(DiscreteEnergyFunctionBase &, std::size_t)>(
                                     [](DiscreteEnergyFunctionBase &self, std::size_t indent) {
                                         const auto str = self.serialize_json().dump(indent);
                                         return str;
                                     }));

    em::class_<Potts, em::base<DiscreteEnergyFunctionBase>>("Potts").constructor<std::size_t, energy_type>();

    em::class_<XArray, em::base<DiscreteEnergyFunctionBase>>("XArray")
        .constructor(&makeXArrayFromList, em::allow_raw_pointers())
        .constructor(&makeXArrayShapeAndTypedArray, em::allow_raw_pointers());

    em::class_<DiscreteConstraintFunctionBase>("DiscreteConstraintFunctionBase");

    em::class_<UniqueLables, em::base<DiscreteConstraintFunctionBase>>("UniqueLables")
        .constructor<std::size_t,         // arity
                     discrete_label_type, // num_labels
                     bool,                // with_ignore_label
                     discrete_label_type, // ignore_label,
                     energy_type          // scale for violation
                     >();
}

} // namespace nxtgm
