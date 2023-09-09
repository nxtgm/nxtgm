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

    template <class T>
    void assign_and_pop(const std::string &key, T &value)
    {
        auto &map = get_map<T>();
        if (auto it = map.find(key); it != map.end())
        {
            value = it->second;
            map.erase(it);
        }
    }

    template <class T, class U>
    void assign_and_pop(const std::string &key, T &value, const U &default_value)
    {
        auto &map = get_map<T>();
        if (auto it = map.find(key); it != map.end())
        {
            value = it->second;
            map.erase(it);
        }
        else
        {
            value = default_value;
        }
    }

  private:
    template <class T, typename std::enable_if<std::is_same<T, std::string>::value, int>::type * = nullptr>
    tsl::ordered_map<std::string, std::string> &get_map()
    {
        return string_parameters;
    }
    template <class T, typename std::enable_if<std::is_same<T, OptimizerParameters>::value, int>::type * = nullptr>
    detail::ordered_map_vec<std::string, OptimizerParameters> &get_map()
    {
        return optimizer_parameters;
    }
    template <class T, typename std::enable_if<std::is_integral<T>::value, int>::type * = nullptr>
    tsl::ordered_map<std::string, int64_t> &get_map()
    {
        return int_parameters;
    }
    template <class T, typename std::enable_if<std::is_floating_point<T>::value, int>::type * = nullptr>
    tsl::ordered_map<std::string, double> &get_map()
    {
        return double_parameters;
    }
};

void ensure_all_handled(const std::string &optimizer_name, const OptimizerParameters &parameters);

// Define the function outside the class definition.
std::ostream &operator<<(std::ostream &out, const OptimizerParameters &p);

} // namespace nxtgm
