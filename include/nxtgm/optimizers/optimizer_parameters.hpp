#pragma once

#include <chrono>
#include <map>
#include <nxtgm/utils/serialize.hpp>
#include <string>
#define NOMINMAX
#include <tsl/ordered_map.h>
#include <vector>
// exceptions
#include <stdexcept>

// std::pair
#include <any>
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

class UnknownParameterException : public std::runtime_error
{
  public:
    UnknownParameterException(const std::string &str);
};

class OptimizerParameters
{
  public:
    class Proxy
    {
      public:
        Proxy(const std::string &key, OptimizerParameters *parent);

        Proxy &operator=(const std::string &value);
        Proxy &operator=(const OptimizerParameters &value);

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
    void assign_and_pop_from_any(const std::string &key, T &value)
    {
        if (auto it = any_parameters.find(key); it != any_parameters.end())
        {
            const std::any &anyval = it->second;
            if (anyval.has_value() == false || anyval.type() != typeid(T))
            {
                throw std::runtime_error(std::string("beliefs_callback must be ") + typeid(T).name() + " but is " +
                                         anyval.type().name());
            }
            else
            {
                value = std::any_cast<T>(anyval);
                any_parameters.erase(it);
            }
        }
    }

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

    void serialize(Serializer &serializer) const;
    static OptimizerParameters deserialize(Deserializer &deserializer);

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

// template<class OS_STREAM>
// OS_STREAM &operator<<(OS_STREAM &out, const OptimizerParameters &p)
// {
//     out << "OptimizerParameters(";
//     bool first = true;
//     for (const auto &[key, value] : p.string_parameters)
//     {
//         if (!first)
//             out << ", ";
//         out << key << "=" << value;
//         first = false;
//     }
//     for (const auto &[key, value] : p.int_parameters)
//     {
//         if (!first)
//             out << ", ";
//         out << key << "=" << value;
//         first = false;
//     }
//     for (const auto &[key, value] : p.double_parameters)
//     {
//         if (!first)
//             out << ", ";
//         out << key << "=" << value;
//         first = false;
//     }
//     for (const auto &[key, value] : p.any_parameters)
//     {
//         if (!first)
//             out << ", ";
//         out << key << "="
//             << "<some any value>";
//         first = false;
//     }
//     for (const auto &[key, value] : p.optimizer_parameters)
//     {
//         if (!first)
//             out << ", ";
//         out << key << "=" << value;
//         first = false;
//     }
//     out << ")";
//     return  out;
// }

} // namespace nxtgm
