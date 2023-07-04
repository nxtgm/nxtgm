#include <tuple>

namespace nxtgm
{
template <typename Tuple, typename F>
void tuple_for_each(Tuple&& tuple, F&& f)
{
    std::apply([&f](auto&&... args) { (f(args), ...); },
               std::forward<Tuple>(tuple));
}

/// \cond
namespace detail
{
template <size_t I, class F, typename... Ts>
typename std::enable_if<I == sizeof...(Ts), void>::type
visit_tuple_breakable(const std::tuple<Ts...>&, F&&)
{
    return;
}
template <size_t I, class F, typename... Ts>
typename std::enable_if<I == sizeof...(Ts), void>::type
visit_tuple_breakable(std::tuple<Ts...>&, F&&)
{
    return;
}

template <size_t I, class F, typename... Ts>
typename std::enable_if<(I < sizeof...(Ts)), void>::type
visit_tuple_breakable(const std::tuple<Ts...>& tup, F&& f)
{
    if (!f(std::get<I>(tup)))
    {
        return;
    }
    // Go to next element
    visit_tuple_breakable<I + 1>(tup, f);
}
template <size_t I, class F, typename... Ts>
typename std::enable_if<(I < sizeof...(Ts)), void>::type
visit_tuple_breakable(std::tuple<Ts...>& tup, F&& f)
{
    if (!f(std::get<I>(tup)))
    {
        return;
    }
    // Go to next element
    visit_tuple_breakable<I + 1>(tup, f);
}
} // namespace detail
/// \endcond

template <typename Tuple, typename F>
void tuple_breakable_for_each(Tuple&& tuple, F&& f)
{
    detail::visit_tuple_breakable<0>(std::forward<Tuple>(tuple),
                                     std::forward<F>(f));
}

} // namespace nxtgm
