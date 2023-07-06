#include <unordered_map>
#include <vector>

namespace nxtgm
{

template <class T>
class SparseArray
{
private:
    using self_type = SparseArray<T>;

    // proxy class do distinguish between read and write access
    class Proxy
    {
    public:
        Proxy(SparseArray<T>& array, std::size_t index)
            : array_(array), index_(index)
        {
        }

        // read access
        operator T() const
        {
            auto it = array_.data_.find(index_);
            if (it == array_.data_.end())
            {
                return T(0);
            }
            else
            {
                return array_.data_.at(index_);
            }
        }

        // write access
        Proxy& operator=(const T& value)
        {
            array_.data_[index_] = value;
            return *this;
        }

    private:
        SparseArray<T>& array_;
        std::size_t index_;
    };

    friend class Proxy;

public:
    template <class ST>
    SparseArray(std::initializer_list<ST> list)
        : size_(1), shape_(list.begin(), list.end()), strides_(shape_.size()),
          data_()
    {
        this->init();
    }

    template <class SHAPE>
    SparseArray(SHAPE&& shape)
        : size_(1), shape_(shape.begin(), shape.end()), strides_(shape_.size()),
          data_()
    {
        this->init();
    }

    std::size_t size() const { return size_; }

    std::size_t num_non_zero_entries() const { return data_.size(); }

    const std::vector<std::size_t>& shape() const { return shape_; }

    std::size_t shape(std::size_t i) const { return shape_[i]; }

    std::size_t dimension() const { return shape_.size(); }

    // read access
    template <class... INDICES>
    T operator()(INDICES... indices) const
    {
        return read_at(variadic_flat_index(indices...));
    }

    // read access
    template <class C>
    T operator[](C&& c) const
    {
        return read_at(flat_index(c));
    }

    // write / read access
    template <class... INDICES>
    Proxy operator()(INDICES... indices)
    {
        return proxy_at(variadic_flat_index(indices...));
    }
    // write / read access
    template <class C>
    Proxy operator[](C&& c)
    {
        return proxy_at(flat_index(c));
    }

    const auto& non_zero_entries() const { return data_; }
    auto& non_zero_entries() { return data_; }

    template <class C>
    void multindex_from_flat_index(std::size_t index, C&& c) const
    {
        for (std::size_t i = 0; i < shape_.size(); ++i)
        {
            c[i] = index / strides_[i];
            index = index % strides_[i];
        }
    }

private:
    // read at flat index
    T read_at(std::size_t index) const
    {
        auto it = data_.find(index);
        if (it == data_.end())
        {
            return T(0);
        }
        else
        {
            return it->second;
        }
    }

    Proxy proxy_at(std::size_t index) { return Proxy(*this, index); }

    void init()
    {
        // compute strides
        strides_.back() = 1;
        for (std::size_t i = shape_.size() - 1; i > 0; --i)
        {
            strides_[i - 1] = strides_[i] * shape_[i];
        }
        // compute size
        size_ = 1;
        for (auto s : shape_)
        {
            size_ *= s;
        }
    }

    template <class... INDICES>
    std::size_t variadic_flat_index(INDICES... indices) const
    {

        constexpr auto n = sizeof...(INDICES);

        if (n == 1)
        {
            // get first index via tuple
            std::size_t index = std::get<0>(std::make_tuple(indices...));

            return index;
        }
        else
        {

            std::size_t index = 0;
            std::size_t i = 0;
            for (auto s : {indices...})
            {
                index += s * strides_[i];
                ++i;
            }
            return index;
        }
    }

    template <class C>
    std::size_t flat_index(C&& c) const
    {
        std::size_t index = 0;
        for (std::size_t i = 0; i < shape_.size(); ++i)
        {
            index += c[i] * strides_[i];
        }
        return index;
    }

    std::size_t size_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::unordered_map<std::size_t, T> data_;
};

} // namespace nxtgm
