#pragma once

#include <nxtgm/utils/random_access_set.hpp>
#include <nxtgm/utils/small_vector.hpp>
#include <xtl/xspan_impl.hpp>

#include <cstdint>
#include <vector>

#include <nxtgm/nxtgm_runtime_checks.hpp>

namespace nxtgm
{
template <typename T>
using span = tcb::span<T>;

using discrete_label_type = std::uint16_t;
using discrete_solution = std::vector<discrete_label_type>;

using const_discrete_solution_span = span<const discrete_label_type>;
using discrete_solution_span = span<discrete_label_type>;

using continuous_label_type = double;
using energy_type = double;

using const_energy_span = span<const energy_type>;
using energy_span = span<energy_type>;

using const_discrete_label_span = span<const discrete_label_type>;
using discrete_label_span = span<discrete_label_type>;

template <typename T, std::size_t N>
using small_vector = SmallVector<T, N>;

template <typename T>
using small_arity_vector = small_vector<T, 10>;

template <typename T>
using small_factor_size_vector = small_vector<T, 100>;

template <typename T>
using flat_set = RandomAccessSet<T>;

constexpr energy_type constraint_feasiblility_limit = 1e-5;

} // namespace nxtgm
