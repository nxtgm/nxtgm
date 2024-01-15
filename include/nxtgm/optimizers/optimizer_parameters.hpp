#pragma once

#include <chrono>
#include <map>
#include <nxtgm/utils/serialize.hpp>
#include <string>
#include <unordered_map>
#include <vector>
// exceptions
#include <stdexcept>

// std::pair
#include <nxtgm/utils/uany.hpp>
#include <utility>

namespace nxtgm
{

template <class K, class V>
using unordered_vector_map = std::unordered_map<K, std::vector<V>>;

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
            parent->int_parameters[key] = std::vector<int64_t>(1, value);
            return *this;
        }

        template <typename T, typename std::enable_if<std::is_floating_point<T>::value, int>::type * = nullptr>
        Proxy &operator=(const T &value)
        {
            parent->double_parameters[key] = std::vector<double>(1, value);
            return *this;
        }

        // push back
        void push_back(const std::string &value);
        void push_back(const OptimizerParameters &value);

        template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type * = nullptr>
        void push_back(const T &value)
        {
            parent->int_parameters[key].push_back(value);
        }

        template <typename T, typename std::enable_if<std::is_floating_point<T>::value, int>::type * = nullptr>
        void push_back(const T &value)
        {
            parent->double_parameters[key].push_back(value);
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

    unordered_vector_map<std::string, std::string> string_parameters;
    unordered_vector_map<std::string, int64_t> int_parameters;
    unordered_vector_map<std::string, double> double_parameters;
    unordered_vector_map<std::string, OptimizerParameters> optimizer_parameters;
    std::unordered_map<std::string, uany> any_parameters;
    bool empty() const;

    template <class T>
    void assign_and_pop_from_any(const std::string &key, T &value)
    {
        if (auto it = any_parameters.find(key); it != any_parameters.end())
        {
            const uany &canyval = it->second;
            // const cast
            uany &anyval = const_cast<uany &>(canyval);
            if (!anyval.has_value())
            {
                throw std::runtime_error(key + std::string(" is empty any val"));
            }
            else
            {
                if (anyval.type() != typeid(T))
                {
                    try
                    {
                        value = uany_cast<T>(anyval);
                        any_parameters.erase(it);
                    }
                    catch (const bad_uany_cast &e)
                    {
                        std::cout << e.what() << std::endl;
                        throw std::runtime_error(key + std::string("is of type ") + typeid(T).name() + " but is " +
                                                 anyval.type().name());
                    }
                }
                else
                {
                    value = uany_cast<T>(anyval);
                    any_parameters.erase(it);
                }
            }
        }
    }

    // vectors:
    // try to assign whats at at the key to the out parameter "value" and pop it from the map
    template <class T>
    void assign_and_pop(const std::string &key, std::vector<T> &value)
    {
        auto &map = get_map<T>();
        if (auto it = map.find(key); it != map.end())
        {
            value = it->second;
            map.erase(it);
        }
    }

    template <class T>
    void assign_and_pop(const std::string &key, std::vector<T> &value, const std::vector<T> &default_value)
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

    // scalars:
    // try to assign whats at at the key to the out parameter "value" and pop it from the map
    // if the key is not found, do nothing
    template <class T>
    void assign_and_pop(const std::string &key, T &value)
    {
        auto &map = get_map<T>();
        if (auto it = map.find(key); it != map.end())
        {
            if (it->second.size() != 1)
            {
                throw std::runtime_error(std::string("assign_and_pop: vector size ") +
                                         std::to_string(it->second.size()) + " != 1");
            }
            value = it->second.front();
            map.erase(it);
        }
    }
    // try to assign whats at at the key to the out parameter "value" and pop it from the map
    // if the key is not found, assign the default value
    template <class T, class U>
    void assign_and_pop(const std::string &key, T &value, const U &default_value)
    {
        auto &map = get_map<T>();
        if (auto it = map.find(key); it != map.end())
        {
            if (it->second.size() != 1)
            {
                throw std::runtime_error(std::string("assign_and_pop: vector size ") +
                                         std::to_string(it->second.size()) + " != 1");
            }
            value = it->second.front();
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
    unordered_vector_map<std::string, std::string> &get_map()
    {
        return string_parameters;
    }
    template <class T, typename std::enable_if<std::is_same<T, OptimizerParameters>::value, int>::type * = nullptr>
    unordered_vector_map<std::string, OptimizerParameters> &get_map()
    {
        return optimizer_parameters;
    }
    template <class T, typename std::enable_if<std::is_integral<T>::value, int>::type * = nullptr>
    unordered_vector_map<std::string, int64_t> &get_map()
    {
        return int_parameters;
    }
    template <class T, typename std::enable_if<std::is_floating_point<T>::value, int>::type * = nullptr>
    unordered_vector_map<std::string, double> &get_map()
    {
        return double_parameters;
    }
};

void ensure_all_handled(const std::string &optimizer_name, const OptimizerParameters &parameters);

} // namespace nxtgm
