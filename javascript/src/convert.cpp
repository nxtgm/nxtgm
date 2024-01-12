#include "convert.hpp"

namespace nxtgm
{
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
} // namespace nxtgm
