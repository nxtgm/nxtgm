#pragma once

#include <algorithm>

namespace nxtghm
{

/// \brief Vector that stores values on the stack if size is smaller than MAX_STACK
/// \tparam T value type
/// \tparam MAX_STACK maximum number of elements kept on the stack
///
/// The member functions resize and clear reduce the size but not the
/// capacity of the vector.
///
/// \ingroup datastructures
template <class T, std::size_t MAX_STACK>
class SmallVector
{
  public:
    typedef T ValueType;
    typedef T value_type;
    typedef T const *ConstIteratorType;
    typedef T const *const_iterator;
    typedef T *IteratorType;
    typedef T *iterator;

    SmallVector();
    SmallVector(const size_t);
    SmallVector(const size_t, const T &);
    SmallVector(const SmallVector<T, MAX_STACK> &);
    ~SmallVector();
    SmallVector<T, MAX_STACK> &operator=(const SmallVector<T, MAX_STACK> &);
    template <class ITERATOR>
    void assign(ITERATOR, ITERATOR);

    size_t size() const;
    T const *begin() const;
    T const *end() const;
    T *const begin();
    T *const end();
    T const &operator[](const size_t) const;
    T &operator[](const size_t);
    void push_back(const T &);
    void resize(const size_t);
    void reserve(const size_t);
    void clear();
    bool empty() const;
    const T &front() const;
    const T &back() const;
    T &front();
    T &back();

  private:
    size_t size_;
    size_t capacity_;
    T stackSequence_[MAX_STACK];
    T *pointerToSequence_;
};

/// constructor
template <class T, size_t MAX_STACK>
SmallVector<T, MAX_STACK>::SmallVector()
    : size_(0),
      capacity_(MAX_STACK),
      pointerToSequence_(stackSequence_)
{
}

/// constructor
/// \param size length of the sequence
template <class T, size_t MAX_STACK>
SmallVector<T, MAX_STACK>::SmallVector(const size_t size)
    : size_(size),
      capacity_(size > MAX_STACK ? size : MAX_STACK)
{
    if (size_ > MAX_STACK)
    {
        pointerToSequence_ = new T[size];
    }
    else
    {
        pointerToSequence_ = stackSequence_;
    }
}

/// constructor
/// \param size lenght of the sequence
/// \param value initial value
template <class T, size_t MAX_STACK>
SmallVector<T, MAX_STACK>::SmallVector(const size_t size, const T &value)
    : size_(size),
      capacity_(size > MAX_STACK ? size : MAX_STACK)
{
    if (size_ > MAX_STACK)
    {
        pointerToSequence_ = new T[size_];
    }
    else
    {
        pointerToSequence_ = stackSequence_;
    }
    std::fill(pointerToSequence_, pointerToSequence_ + size_, value);
}

/// copy constructor
/// \param other container to copy
template <class T, size_t MAX_STACK>
SmallVector<T, MAX_STACK>::SmallVector(const SmallVector<T, MAX_STACK> &other)
    : size_(other.size_),
      capacity_(other.capacity_)
{
    if (size_ > MAX_STACK)
    {
        pointerToSequence_ = new T[size_];
    }
    else
    {
        pointerToSequence_ = stackSequence_;
    }
    std::copy(other.pointerToSequence_, other.pointerToSequence_ + size_, pointerToSequence_);
}

/// destructor
template <class T, size_t MAX_STACK>
SmallVector<T, MAX_STACK>::~SmallVector()
{
    if (capacity_ > MAX_STACK)
    {
        delete[] pointerToSequence_;
    }
}

/// assignment operator
/// \param other container to copy
template <class T, size_t MAX_STACK>
SmallVector<T, MAX_STACK> &SmallVector<T, MAX_STACK>::operator=(const SmallVector<T, MAX_STACK> &other)
{
    if (&other != this)
    {
        if (other.size_ > MAX_STACK)
        {
            // delete old sequence
            if (size_ > MAX_STACK)
            {
                delete[] pointerToSequence_;
                pointerToSequence_ = new T[other.size_];
            }
            // nothing to delete
            else
            {
                pointerToSequence_ = new T[other.size_];
            }
            size_ = other.size_;
            capacity_ = size_;
        }
        else
        {
            pointerToSequence_ = stackSequence_;
            size_ = other.size_;
        }
        std::copy(other.pointerToSequence_, other.pointerToSequence_ + size_, pointerToSequence_);
    }
    return *this;
}

/// size
template <class T, size_t MAX_STACK>
inline size_t SmallVector<T, MAX_STACK>::size() const
{
    return size_;
}

/// begin iterator
template <class T, size_t MAX_STACK>
inline T const *SmallVector<T, MAX_STACK>::begin() const
{
    return pointerToSequence_;
}

/// end iterator
template <class T, size_t MAX_STACK>
inline T const *SmallVector<T, MAX_STACK>::end() const
{
    return pointerToSequence_ + size_;
}

/// begin iterator
template <class T, size_t MAX_STACK>
inline T *const SmallVector<T, MAX_STACK>::begin()
{
    return pointerToSequence_;
}

/// end iterator
template <class T, size_t MAX_STACK>
inline T *const SmallVector<T, MAX_STACK>::end()
{
    return pointerToSequence_ + size_;
}

/// access entries
template <class T, size_t MAX_STACK>
inline T const &SmallVector<T, MAX_STACK>::operator[](const size_t index) const
{
    return pointerToSequence_[index];
}

/// access entries
template <class T, size_t MAX_STACK>
inline T &SmallVector<T, MAX_STACK>::operator[](const size_t index)
{
    return pointerToSequence_[index];
}

/// append a value
/// \param value value to append
template <class T, size_t MAX_STACK>
inline void SmallVector<T, MAX_STACK>::push_back(const T &value)
{
    if (capacity_ == size_)
    {
        T *tmp = new T[capacity_ * 2];
        std::copy(pointerToSequence_, pointerToSequence_ + size_, tmp);
        if (capacity_ > MAX_STACK)
        {
            delete[] pointerToSequence_;
        }
        capacity_ *= 2;
        pointerToSequence_ = tmp;
    }
    pointerToSequence_[size_] = value;
    ++size_;
}

/// resize the sequence
/// \param size new size of the container
template <class T, size_t MAX_STACK>
inline void SmallVector<T, MAX_STACK>::resize(const size_t size)
{
    if (size > capacity_)
    {
        T *tmp = new T[size];
        std::copy(pointerToSequence_, pointerToSequence_ + size_, tmp);
        if (capacity_ > MAX_STACK)
        {
            delete[] pointerToSequence_;
        }
        capacity_ = size;
        pointerToSequence_ = tmp;
    }
    size_ = size;
}

/// reserve memory
/// \param  size new size of the container
template <class T, size_t MAX_STACK>
inline void SmallVector<T, MAX_STACK>::reserve(const size_t size)
{
    if (size > capacity_)
    {
        T *tmp = new T[size];
        std::copy(pointerToSequence_, pointerToSequence_ + size_, tmp);
        if (capacity_ > MAX_STACK)
        {
            delete[] pointerToSequence_;
        }
        capacity_ = size;
        pointerToSequence_ = tmp;
    }
}

/// clear the sequence
template <class T, size_t MAX_STACK>
inline void SmallVector<T, MAX_STACK>::clear()
{
    if (capacity_ > MAX_STACK)
    {
        delete[] pointerToSequence_;
    }
    pointerToSequence_ = stackSequence_;
    capacity_ = MAX_STACK;
    size_ = 0;
}

/// query if the sequence is empty
template <class T, size_t MAX_STACK>
inline bool SmallVector<T, MAX_STACK>::empty() const
{
    return (size_ == 0);
}

/// assign values
/// \param begin begin iterator
/// \param end end iterator
template <class T, size_t MAX_STACK>
template <class ITERATOR>
inline void SmallVector<T, MAX_STACK>::assign(ITERATOR begin, ITERATOR end)
{
    this->resize(std::distance(begin, end));
    std::copy(begin, end, pointerToSequence_);
}

/// reference to the last entry
template <class T, size_t MAX_STACK>
inline const T &SmallVector<T, MAX_STACK>::back() const
{
    return pointerToSequence_[size_ - 1];
}

/// reference to the last entry
template <class T, size_t MAX_STACK>
inline T &SmallVector<T, MAX_STACK>::back()
{
    return pointerToSequence_[size_ - 1];
}

/// reference to the first entry
template <class T, size_t MAX_STACK>
inline const T &SmallVector<T, MAX_STACK>::front() const
{
    return pointerToSequence_[0];
}

/// reference to the first entry
template <class T, size_t MAX_STACK>
inline T &SmallVector<T, MAX_STACK>::front()
{
    return pointerToSequence_[0];
}

} // namespace nxtghm
