#pragma once

#include <chrono>
#include <map>
#include <string>
#include <tsl/ordered_map.h>
#include <vector>

// std::pair
#include <any>
#include <iostream>
#include <utility>

namespace nxtgm
{
/// \cond
namespace detail
{
template <typename KEY, typename VALUE>
using ordered_map_vec =
    tsl::ordered_map<KEY, VALUE, std::hash<KEY>, std::equal_to<KEY>, std::allocator<std::pair<KEY, VALUE>>,
                     std::vector<std::pair<KEY, VALUE>>, std::uint_least32_t>;
}
/// \endcond

class OptimizerParameters
{
  public:
    class Proxy
    {
      public:
        Proxy(const std::string &key, OptimizerParameters *parent);

        Proxy &operator=(const std::string &value);
        Proxy &operator=(const OptimizerParameters &value);

        // operator for all floating point types (but in a way that multiple overloads can be defined)
        template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type * = nullptr>
        Proxy &operator=(const T &value)
        {
            parent->int_parameters[key] = value;
            return *this;
        }

        template <typename T, typename std::enable_if<std::is_floating_point<T>::value, int>::type * = nullptr>
        Proxy &operator=(const T &value)
        {
            parent->double_parameters[key] = value;
            return *this;
        }

      private:
        std::string key;
        OptimizerParameters *parent;
    };

    OptimizerParameters() = default;
    OptimizerParameters(const OptimizerParameters &) = default;
    OptimizerParameters(OptimizerParameters &&) = default;
    OptimizerParameters &operator=(const OptimizerParameters &) = default;
    OptimizerParameters &operator=(OptimizerParameters &&) = default;
    ~OptimizerParameters() = default;

    // write access
    Proxy operator[](const std::string &key);

    tsl::ordered_map<std::string, std::string> string_parameters;
    tsl::ordered_map<std::string, int64_t> int_parameters;
    tsl::ordered_map<std::string, double> double_parameters;
    tsl::ordered_map<std::string, std::any> any_parameters;
    detail::ordered_map_vec<std::string, OptimizerParameters> optimizer_parameters;
    bool empty() const;
};

void ensure_all_handled(const std::string &optimizer_name, const OptimizerParameters &parameters);

// Define the function outside the class definition.
std::ostream &operator<<(std::ostream &out, const OptimizerParameters &p);

} // namespace nxtgm
