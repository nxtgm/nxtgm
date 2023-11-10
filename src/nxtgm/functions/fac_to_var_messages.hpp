#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

namespace nxtgm
{

template <class VT>
inline void simple_second_order_fac_to_var_messages(const VT *vt, const discrete_label_type nl,
                                                    const energy_type **inMsgs, energy_type **outMsgs)
{

    // initialize
    for (auto a = 0; a < 2; ++a)
        std::fill(outMsgs[a], outMsgs[a] + nl, std::numeric_limits<energy_type>::infinity());
    // minimize
    for (discrete_label_type l1 = 0; l1 < nl; ++l1)
        for (discrete_label_type l0 = 0; l0 < nl; ++l0)
        {
            const energy_type facVal = vt->value(l0, l1);
            outMsgs[0][l0] = std::min(outMsgs[0][l0], facVal + inMsgs[1][l1]);
            outMsgs[1][l1] = std::min(outMsgs[1][l1], facVal + inMsgs[0][l0]);
        }
}

template <class VT>
inline void l1_fac_to_var_messge(const VT *vt, const discrete_label_type nl, const energy_type beta,
                                 const energy_type **inMsgs, energy_type **outMsgs)
{
    if (beta >= 0)
    {
        std::copy(inMsgs[1], inMsgs[1] + nl, outMsgs[0]);
        std::copy(inMsgs[0], inMsgs[0] + nl, outMsgs[1]);
        // "forward pass"
        for (discrete_label_type l = 1; l < nl; ++l)
        {
            outMsgs[0][l] = std::min(outMsgs[0][l], outMsgs[0][l - 1] + beta);
            outMsgs[1][l] = std::min(outMsgs[1][l], outMsgs[1][l - 1] + beta);
        }
        // backward pass
        for (discrete_label_type l = nl - 2; l >= 0; --l)
        {
            outMsgs[0][l] = std::min(outMsgs[0][l], outMsgs[0][l + 1] + beta);
            outMsgs[1][l] = std::min(outMsgs[1][l], outMsgs[1][l + 1] + beta);
            if (l == 0)
                break;
        }
    }
    else
    {
        simple_second_order_fac_to_var_messages(vt, nl, inMsgs, outMsgs);
    }
}

template <class VT>
inline void truncatedl1_fac_to_var_messge(const VT *vt, const discrete_label_type nl, const energy_type beta,
                                          const energy_type truncateAt, const energy_type **inMsgs,
                                          energy_type **outMsgs)
{
    if (beta >= 0)
    {
        l1_fac_to_var_messge(vt, nl, beta, inMsgs, outMsgs);

        const energy_type minIn0Trunc = *std::min_element(inMsgs[0], inMsgs[0] + nl) + truncateAt;
        const energy_type minIn1Trunc = *std::min_element(inMsgs[1], inMsgs[1] + nl) + truncateAt;

        for (discrete_label_type l = 0; l < nl; ++l)
        {
            outMsgs[0][l] = std::min(outMsgs[0][l], minIn1Trunc);
            outMsgs[1][l] = std::min(outMsgs[1][l], minIn0Trunc);
        }
    }
    else
    {
        simple_second_order_fac_to_var_messages(vt, nl, inMsgs, outMsgs);
    }
}

} // namespace nxtgm
