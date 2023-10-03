#pragma once
#include <emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/val.h>

namespace nxtgm
{

// helper struct to convert type T to string like "Int8Array" for int8_t and so on
template <typename T>
struct type_to_string;

// macro for that
#define TYPE_TO_STRING(TYPE, NAME)                                                                                     \
    template <>                                                                                                        \
    struct type_to_string<TYPE>                                                                                        \
    {                                                                                                                  \
        static std::string name()                                                                                      \
        {                                                                                                              \
            return #NAME;                                                                                              \
        }                                                                                                              \
    }

// specializations for all types
TYPE_TO_STRING(int8_t, Int8Array);
TYPE_TO_STRING(int16_t, Int16Array);
TYPE_TO_STRING(int32_t, Int32Array);
TYPE_TO_STRING(int64_t, BigInt64Array);
TYPE_TO_STRING(uint8_t, Uint8Array);
TYPE_TO_STRING(uint16_t, Uint16Array);
TYPE_TO_STRING(uint32_t, Uint32Array);
TYPE_TO_STRING(uint64_t, BigUint64Array);
TYPE_TO_STRING(float, Float32Array);
TYPE_TO_STRING(double, Float64Array);

template <class T>
emscripten::val ptr_range_to_typed_array(const T *begin, std::size_t size)
{
    emscripten::val mem_view = emscripten::val(emscripten::typed_memory_view(size, begin));
    emscripten::val mem_copy = emscripten::val::global(type_to_string<T>::name().c_str()).new_(mem_view);
    return mem_copy;
}

std::size_t get_length(const emscripten::val &value)
{
    return value["length"].as<std::size_t>();
}

void require_array(const emscripten::val &value)
{
    if (!value.isArray())
    {
        throw std::runtime_error("value is not an array");
    }
}
void require_number(const emscripten::val &value)
{
    if (!value.isNumber())
    {
        throw std::runtime_error("value is not an number");
    }
}
template <class T>
T as_number_checked(const emscripten::val &value)
{
    require_number(value);
    return value.as<T>();
}

template <class T>
std::vector<T> as_vector_of_numbers_checked(const emscripten::val &value)
{
    require_array(value);
    const auto length = get_length(value);
    std::vector<T> result;
    result.reserve(length);
    for (int i = 0; i < length; i++)
    {
        result.push_back(as_number_checked<T>(value[i]));
    }
    return result;
}

} // namespace nxtgm
