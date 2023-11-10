#include "argmin2.hpp"

namespace nxtgm
{
std::pair<std::size_t, std::size_t> arg2min(const energy_type *begin, const energy_type *end)
{
    std::pair<std::size_t, std::size_t> res;
    energy_type min0 = std::numeric_limits<energy_type>::infinity();
    energy_type min1 = min0;
    const auto dist = std::distance(begin, end);
    for (auto i = 0; i < dist; ++i)
    {
        const energy_type val = begin[i];
        if (val < min1)
        {
            if (val < min0)
            {

                min1 = min0;
                res.second = res.first;
                min0 = val;
                res.first = i;
                continue;
            }
            else
            {
                min1 = val;
                res.second = i;
            }
        }
    }
    return res;
}
} // namespace nxtgm
