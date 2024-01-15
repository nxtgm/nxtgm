
#pragma once

// xtensor view
#include <xtensor/xview.hpp>
// xtensor adapt
#include <xtensor/xadapt.hpp>

namespace nxtgm
{

template <class XTENSOR>
auto bind_at(const XTENSOR &xtensor, std::size_t axis, std::size_t value)
{
    xt::xstrided_slice_vector sv(xtensor.dimension(), xt::all());
    sv[axis] = value;
    return xt::strided_view(xtensor, sv);
}

template <class XTENSOR, class AXES, class LABELS>
auto bind_many(const XTENSOR &xtensor, AXES &&axes, LABELS &&labels)
{
    xt::xstrided_slice_vector sv(xtensor.dimension(), xt::all());

    for (std::size_t i = 0; i < axes.size(); ++i)
    {
        sv[axes[i]] = labels[i];
    }

    return xt::strided_view(xtensor, sv);
}

} // namespace nxtgm
