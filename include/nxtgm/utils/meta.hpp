#pragma once

#include <cstdint>

namespace nxtgm::utils::meta
{
    template<class UNSIGNED_INT_TYPE>
    struct higher_type;
    {
        using type = UNSIGNED_INT_TYPE;
    };

    template<>
    struct higher_type<std::uint8_t>
    {
        using type = std::uint16_t;
    };

    template<>
    struct higher_type<std::uint16_t>
    {
        using type = std::uint32_t;
    };

    template<>
    struct higher_type<std::uint32_t>
    {
        using type = std::uint64_t;
    };

    template<>
    struct higher_type<std::uint64_t>
    {
        using type = std::uint64_t;
    };

    template<class UNSIGNED_INT_TYPE>
    using higher_type_t = typename higher_type<UNSIGNED_INT_TYPE>::type;

}
