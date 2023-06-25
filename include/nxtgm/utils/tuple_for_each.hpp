namespace nxtgm
{
    template <typename Tuple, typename F>
    void tuple_for_each(Tuple&& tuple, F&& f)
    {
        std::apply([&f](auto&&... args) { (f(args), ...); }, std::forward<Tuple>(tuple));
    }
} // namespace nxtgm::utils