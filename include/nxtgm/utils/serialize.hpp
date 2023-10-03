#pragma once

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>

namespace nxtgm
{
class Serializer
{
  public:
    Serializer(std::ostream &os)
        : os_(os)
    {
    }

    template <class T>
    void operator()(const T &value)
    {
        os_.write((const char *)&value, sizeof(value));
    }
    void operator()(const std::string &value)
    {
        this->operator()(value.data(), value.size());
    }
    template <class T>
    void operator()(const T *begin, uint64_t size)
    {
        this->operator()(size);
        os_.write((const char *)begin, sizeof(T) * size);
    }

    template <class T>
    void operator()(const std::vector<T> &value)
    {
        this->operator()(value.data(), value.size());
    }

    template <class T>
    void operator()(const xt::xarray<T> &value)
    {
        op_xarray_like(value);
    }
    template <class T, std::size_t DIM>
    void operator()(const xt::xtensor<T, DIM> &value)
    {
        op_xarray_like(value);
    }
    template <class K, class V>
    void operator()(const std::unordered_map<K, V> &value)
    {
        op_map_like(value);
    }

  private:
    template <class ARR>
    void op_xarray_like(const ARR &value)
    {
        const auto &shape = value.shape();
        this->operator()(shape.data(), shape.size());
        this->operator()(value.data(), value.size());
    }
    template <class MAP>
    void op_map_like(const MAP &value)
    {
        this->operator()(value.size());
        for (const auto &[key, val] : value)
        {
            this->operator()(key);
            this->operator()(val);
        }
    }

    std::ostream &os_;
};

class Deserializer
{
  public:
    Deserializer(std::istream &is)
        : is_(is)
    {
    }

    template <class T>
    void operator()(T &value)
    {
        is_.read((char *)&value, sizeof(value));
    }
    template <class T>
    void operator()(T *begin, std::size_t size)
    {
        is_.read((char *)begin, sizeof(T) * size);
    }
    void operator()(std::string &value)
    {
        uint64_t size;
        this->operator()(size);
        value.resize(size);
        this->operator()(value.data(), size);
    }

    template <class T>
    void operator()(std::vector<T> &value)
    {
        uint64_t size;
        this->operator()(size);
        value.resize(size);
        this->operator()(value.data(), size);
    }

    template <class T>
    void operator()(xt::xarray<T> &value)
    {
        using shape_value_type = typename xt::xarray<T>::shape_type::value_type;
        uint64_t dimension;
        this->operator()(dimension);

        std::vector<shape_value_type> shape(dimension);
        this->operator()(shape.data(), dimension);
        for (auto &v : shape)
        {
            std::cout << v << " ";
        }
        uint64_t size;
        this->operator()(size);
        std::cout << std::endl;
        value.resize(shape);
        this->operator()(value.data(), value.size());
    }

    template <class K, class V>
    void operator()(std::unordered_map<K, V> &value)
    {
        op_map_like(value);
    }

  private:
    template <class MAP>
    void op_map_like(MAP &value)
    {
        using key_type = typename MAP::key_type;
        using mapped_type = typename MAP::mapped_type;

        uint64_t size;
        this->operator()(size);
        for (std::size_t i = 0; i < size; ++i)
        {
            key_type key;
            mapped_type val;

            this->operator()(key);
            this->operator()(val);
            value[key] = val;
        }
    }

    std::istream &is_;
};
} // namespace nxtgm
