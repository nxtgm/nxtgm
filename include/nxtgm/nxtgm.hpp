#pragma once

#include <boost/container/small_vector.hpp>
#include <boost/core/span.hpp>
#include <cstdint>

#include <vector>

namespace nxtgm
{
template <typename T>
using span = boost::span<T>;

using discrete_label_type = std::uint16_t;
using discrete_solution = std::vector<discrete_label_type>;
using continuous_label_type = double;
using energy_type = double;

using const_energy_span = span<const energy_type>;
using energy_span = span<energy_type>;

using const_discrete_label_span = span<const discrete_label_type>;
using discrete_label_span = span<discrete_label_type>;

template <typename T>
using small_arity_vector = boost::container::small_vector<T, 4>;

constexpr energy_type constraint_feasiblility_limit = 1e-5;

} // namespace nxtgm
