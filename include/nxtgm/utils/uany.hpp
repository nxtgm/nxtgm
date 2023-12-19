#pragma once
#include <memory>
#include <typeinfo>
#include <utility>

namespace nxtgm
{

template <typename T>
struct is_in_place_type : std::false_type
{
};

template <typename T>
struct is_in_place_type<std::in_place_type_t<T>> : std::true_type
{
};

class uany
{
    template <typename ValueType>
    friend const ValueType *uany_cast(const uany *) noexcept;

    template <typename ValueType>
    friend ValueType *uany_cast(uany *) noexcept;

  public:
    // constructors

    constexpr uany() noexcept = default;

    uany(const uany &other)
    {
        if (other.instance)
        {
            instance = other.instance->clone();
        }
    }

    uany(uany &&other) noexcept
        : instance(std::move(other.instance))
    {
    }

    template <typename ValueType, typename = std::enable_if_t<!std::is_same_v<std::decay_t<ValueType>, uany> &&
                                                              !is_in_place_type<std::decay_t<ValueType>>::value &&
                                                              std::is_copy_constructible_v<std::decay_t<ValueType>>>>
    uany(ValueType &&value)
    {
        static_assert(std::is_copy_constructible_v<std::decay_t<ValueType>>, "program is ill-formed");
        emplace<std::decay_t<ValueType>>(std::forward<ValueType>(value));
    }

    template <typename ValueType, typename... Args,
              typename = std::enable_if_t<std::is_constructible_v<std::decay_t<ValueType>, Args...> &&
                                          std::is_copy_constructible_v<std::decay_t<ValueType>>>>
    explicit uany(std::in_place_type_t<ValueType>, Args &&...args)
    {
        emplace<std::decay_t<ValueType>>(std::forward<Args>(args)...);
    }

    template <typename ValueType, typename List, typename... Args,
              typename = std::enable_if_t<
                  std::is_constructible_v<std::decay_t<ValueType>, std::initializer_list<List> &, Args...> &&
                  std::is_copy_constructible_v<std::decay_t<ValueType>>>>
    explicit uany(std::in_place_type_t<ValueType>, std::initializer_list<List> list, Args &&...args)
    {
        emplace<std::decay_t<ValueType>>(list, std::forward<Args>(args)...);
    }

    // assignment operators

    uany &operator=(const uany &rhs)
    {
        uany(rhs).swap(*this);
        return *this;
    }

    uany &operator=(uany &&rhs) noexcept
    {
        uany(std::move(rhs)).swap(*this);
        return *this;
    }

    template <typename ValueType>
    std::enable_if_t<
        !std::is_same_v<std::decay_t<ValueType>, uany> && std::is_copy_constructible_v<std::decay_t<ValueType>>, uany &>
    operator=(ValueType &&rhs)
    {
        uany(std::forward<ValueType>(rhs)).swap(*this);
        return *this;
    }

    // modifiers

    template <typename ValueType, typename... Args>
    std::enable_if_t<std::is_constructible_v<std::decay_t<ValueType>, Args...> &&
                         std::is_copy_constructible_v<std::decay_t<ValueType>>,
                     std::decay_t<ValueType> &> inline emplace(Args &&...args)
    {
        auto new_inst = std::make_unique<storage_impl<std::decay_t<ValueType>>>(std::forward<Args>(args)...);
        std::decay_t<ValueType> &value = new_inst->value;
        instance = std::move(new_inst);
        return value;
    }

    template <typename ValueType, typename List, typename... Args>
    std::enable_if_t<std::is_constructible_v<std::decay_t<ValueType>, std::initializer_list<List> &, Args...> &&
                         std::is_copy_constructible_v<std::decay_t<ValueType>>,
                     std::decay_t<ValueType> &> inline emplace(std::initializer_list<List> list, Args &&...args)
    {
        auto new_inst = std::make_unique<storage_impl<std::decay_t<ValueType>>>(list, std::forward<Args>(args)...);
        std::decay_t<ValueType> &value = new_inst->value;
        instance = std::move(new_inst);
        return value;
    }

    inline void reset() noexcept
    {
        instance.reset();
    }

    void swap(uany &other) noexcept
    {
        std::swap(instance, other.instance);
    }

    // observers

    bool has_value() const noexcept
    {
        return static_cast<bool>(instance);
    }

    const std::type_info &type() const noexcept
    {
        return instance ? instance->type() : typeid(void);
    }

  private:
    struct storage_base;

    std::unique_ptr<storage_base> instance;

    struct storage_base
    {
        virtual ~storage_base() = default;

        virtual const std::type_info &type() const noexcept = 0;
        virtual std::unique_ptr<storage_base> clone() const = 0;
    };

    template <typename ValueType>
    struct storage_impl final : public storage_base
    {
        template <typename... Args>
        storage_impl(Args &&...args)
            : value(std::forward<Args>(args)...)
        {
        }

        const std::type_info &type() const noexcept override
        {
            return typeid(ValueType);
        }

        std::unique_ptr<storage_base> clone() const override
        {
            return std::make_unique<storage_impl<ValueType>>(value);
        }

        ValueType value;
    };
};

} // namespace nxtgm

template <>
inline void std::swap(nxtgm::uany &lhs, nxtgm::uany &rhs) noexcept
{
    lhs.swap(rhs);
}

namespace nxtgm
{

class bad_uany_cast : public std::exception
{
  public:
    inline const char *what() const noexcept
    {
        return "bad uany cast";
    }
};

// C++20
template <typename T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

// uany_cast

template <typename ValueType>
inline ValueType uany_cast(const uany &anything)
{
    using value_type_cvref = remove_cvref_t<ValueType>;
    static_assert(std::is_constructible_v<ValueType, const value_type_cvref &>, "program is ill-formed");
    if (auto *value = uany_cast<value_type_cvref>(&anything))
    {
        return static_cast<ValueType>(*value);
    }
    else
    {
        throw bad_uany_cast();
    }
}

template <typename ValueType>
inline ValueType uany_cast(uany &anything)
{
    using value_type_cvref = remove_cvref_t<ValueType>;
    static_assert(std::is_constructible_v<ValueType, value_type_cvref &>, "program is ill-formed");
    if (auto *value = uany_cast<value_type_cvref>(&anything))
    {
        return static_cast<ValueType>(*value);
    }
    else
    {
        throw bad_uany_cast();
    }
}

template <typename ValueType>
inline ValueType uany_cast(uany &&anything)
{
    using value_type_cvref = remove_cvref_t<ValueType>;
    static_assert(std::is_constructible_v<ValueType, value_type_cvref>, "program is ill-formed");
    if (auto *value = uany_cast<value_type_cvref>(&anything))
    {
        return static_cast<ValueType>(std::move(*value));
    }
    else
    {
        throw bad_uany_cast();
    }
}

template <typename ValueType>
inline const ValueType *uany_cast(const uany *anything) noexcept
{
    if (!anything)
        return nullptr;
    auto *storage = dynamic_cast<uany::storage_impl<ValueType> *>(anything->instance.get());
    if (!storage)
        return nullptr;
    return &storage->value;
}

template <typename ValueType>
inline ValueType *uany_cast(uany *anything) noexcept
{
    return const_cast<ValueType *>(uany_cast<ValueType>(static_cast<const uany *>(anything)));
}

// make_any

template <typename ValueType, typename... Args>
inline uany make_any(Args &&...args)
{
    return uany(std::in_place_type<ValueType>, std::forward<Args>(args)...);
}

template <typename ValueType, typename List, typename... Args>
inline uany make_any(std::initializer_list<List> list, Args &&...args)
{
    return uany(std::in_place_type<ValueType>, list, std::forward<Args>(args)...);
}

} // namespace nxtgm
