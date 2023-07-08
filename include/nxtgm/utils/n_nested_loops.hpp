#pragma once

namespace nxtgm
{

template <class INDEX_TYPE, class SHAPE, class BUFFER, class F>
void exitable_n_nested_loops_fallback(const std::size_t n, SHAPE&& shape,
                                      BUFFER&& buffer, F&& f)
{
    // initialize the solution with zeros
    for (INDEX_TYPE i = 0; i < n; ++i)
    {
        buffer[i] = 0;
    }

    const auto last_var = n - 1;
    auto index = last_var;
    while (true)
    {
        // TODO: Your inner loop code goes here. You can inspect the values in
        // slots
        if (!f(buffer))
        {
            return;
        }

        // Increment
        buffer[last_var]++;

        // Carry
        while (buffer[index] == shape[index])
        {
            // Overflow, we're done
            if (index == 0)
            {
                return;
            }

            buffer[index--] = 0;
            buffer[index]++;
        }
        index = last_var;
    }
}

template <class INDEX_TYPE, class SHAPE, class BUFFER, class F>
void exitable_n_nested_loops(const std::size_t n, SHAPE&& shape,
                             BUFFER&& buffer, F&& f)
{
    switch (n)
    {
        case 0:
            return;
        case 1:
            for (INDEX_TYPE i = 0; i < shape[0]; ++i)
            {
                buffer[0] = i;
                if (!f(buffer))
                    return;
            }
            return;
        case 2:
            for (INDEX_TYPE i = 0; i < shape[0]; ++i)
            {
                buffer[0] = i;
                for (INDEX_TYPE j = 0; j < shape[1]; ++j)
                {
                    buffer[1] = j;
                    if (!f(buffer))
                        return;
                }
            }
            return;
        case 3:
            for (INDEX_TYPE i = 0; i < shape[0]; ++i)
            {
                buffer[0] = i;
                for (INDEX_TYPE j = 0; j < shape[1]; ++j)
                {
                    buffer[1] = j;
                    for (INDEX_TYPE k = 0; k < shape[2]; ++k)
                    {
                        buffer[2] = k;
                        if (!f(buffer))
                            return;
                    }
                }
            }
            return;
        case 4:
            for (INDEX_TYPE i = 0; i < shape[0]; ++i)
            {
                buffer[0] = i;
                for (INDEX_TYPE j = 0; j < shape[1]; ++j)
                {
                    buffer[1] = j;
                    for (INDEX_TYPE k = 0; k < shape[2]; ++k)
                    {
                        buffer[2] = k;
                        for (INDEX_TYPE l = 0; l < shape[3]; ++l)
                        {
                            buffer[3] = l;
                            if (!f(buffer))
                                return;
                        }
                    }
                }
            }
            return;
        default:
            exitable_n_nested_loops_fallback<INDEX_TYPE>(
                n, std::forward<SHAPE>(shape), std::forward<BUFFER>(buffer),
                std::forward<F>(f));
            return;
    }
}

template <class INDEX_TYPE, class SHAPE, class BUFFER, class F>
void n_nested_loops(const std::size_t n, SHAPE&& shape, BUFFER& buffer, F&& f)
{
    exitable_n_nested_loops<INDEX_TYPE>(n, std::forward<SHAPE>(shape),
                                        std::forward<BUFFER>(buffer),
                                        [&](BUFFER& buffer)
                                        {
                                            f(buffer);
                                            return true;
                                        });
}

template <class INDEX_TYPE, class SHAPE, class BUFFER, class F>
void exitable_n_nested_loop_unique_labels_fallback(const std::size_t n,
                                                   SHAPE&& shape,
                                                   BUFFER&& buffer, F&& f)
{
    throw std::runtime_error("Not implemented yet for n > 10");
}

template <class INDEX_TYPE, class SHAPE, class BUFFER, class F>
void exitable_n_nested_loop_unique_labels(const std::size_t n, SHAPE&& shape,
                                          BUFFER&& buffer, F&& f)
{

    switch (n)
    {
        case 1:
        {
            for (INDEX_TYPE x0 = 0; x0 < shape[0]; ++x0)
            {
                buffer[0] = x0;
                if (!f(buffer))
                {
                    return;
                }
            }
        }
        case 2:
        {
            for (INDEX_TYPE x0 = 0; x0 < shape[0]; ++x0)
            {
                buffer[0] = x0;
                for (INDEX_TYPE x1 = 0; x1 < shape[1]; ++x1)
                {
                    if (x0 != x1)
                    {
                        buffer[1] = x1;
                        if (!f(buffer))
                        {
                            return;
                        }
                    }
                }
            }
        }
        case 3:
        {
            for (INDEX_TYPE x0 = 0; x0 < shape[0]; ++x0)
            {
                buffer[0] = x0;
                for (INDEX_TYPE x1 = 0; x1 < shape[1]; ++x1)
                {
                    if (x0 != x1)
                    {
                        buffer[1] = x1;
                        for (INDEX_TYPE x2 = 0; x2 < shape[2]; ++x2)
                        {
                            if (x0 != x2 && x1 != x2)
                            {
                                buffer[2] = x2;
                                if (!f(buffer))
                                {
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        }
        case 4:
        {
            for (INDEX_TYPE x0 = 0; x0 < shape[0]; ++x0)
            {
                buffer[0] = x0;
                for (INDEX_TYPE x1 = 0; x1 < shape[1]; ++x1)
                {
                    if (x0 != x1)
                    {
                        buffer[1] = x1;
                        for (INDEX_TYPE x2 = 0; x2 < shape[2]; ++x2)
                        {
                            if (x0 != x2 && x1 != x2)
                            {
                                buffer[2] = x2;
                                for (INDEX_TYPE x3 = 0; x3 < shape[3]; ++x3)
                                {
                                    if (x0 != x3 && x1 != x3 && x2 != x3)
                                    {
                                        buffer[3] = x3;
                                        if (!f(buffer))
                                        {
                                            return;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        case 5:
        {
            for (INDEX_TYPE x0 = 0; x0 < shape[0]; ++x0)
            {
                buffer[0] = x0;
                for (INDEX_TYPE x1 = 0; x1 < shape[1]; ++x1)
                {
                    if (x0 != x1)
                    {
                        buffer[1] = x1;
                        for (INDEX_TYPE x2 = 0; x2 < shape[2]; ++x2)
                        {
                            if (x0 != x2 && x1 != x2)
                            {
                                buffer[2] = x2;
                                for (INDEX_TYPE x3 = 0; x3 < shape[3]; ++x3)
                                {
                                    if (x0 != x3 && x1 != x3 && x2 != x3)
                                    {
                                        buffer[3] = x3;
                                        for (INDEX_TYPE x4 = 0; x4 < shape[4];
                                             ++x4)
                                        {
                                            if (x0 != x4 && x1 != x4 &&
                                                x2 != x4 && x3 != x4)
                                            {
                                                buffer[4] = x4;
                                                if (!f(buffer))
                                                {
                                                    return;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        case 6:
        {
            for (INDEX_TYPE x0 = 0; x0 < shape[0]; ++x0)
            {
                buffer[0] = x0;
                for (INDEX_TYPE x1 = 0; x1 < shape[1]; ++x1)
                {
                    if (x0 != x1)
                    {
                        buffer[1] = x1;
                        for (INDEX_TYPE x2 = 0; x2 < shape[2]; ++x2)
                        {
                            if (x0 != x2 && x1 != x2)
                            {
                                buffer[2] = x2;
                                for (INDEX_TYPE x3 = 0; x3 < shape[3]; ++x3)
                                {
                                    if (x0 != x3 && x1 != x3 && x2 != x3)
                                    {
                                        buffer[3] = x3;
                                        for (INDEX_TYPE x4 = 0; x4 < shape[4];
                                             ++x4)
                                        {
                                            if (x0 != x4 && x1 != x4 &&
                                                x2 != x4 && x3 != x4)
                                            {
                                                buffer[4] = x4;
                                                for (INDEX_TYPE x5 = 0;
                                                     x5 < shape[5]; ++x5)
                                                {
                                                    if (x0 != x5 && x1 != x5 &&
                                                        x2 != x5 && x3 != x5 &&
                                                        x4 != x5)
                                                    {
                                                        buffer[5] = x5;
                                                        if (!f(buffer))
                                                        {
                                                            return;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        case 7:
        {
            for (INDEX_TYPE x0 = 0; x0 < shape[0]; ++x0)
            {
                buffer[0] = x0;
                for (INDEX_TYPE x1 = 0; x1 < shape[1]; ++x1)
                {
                    if (x0 != x1)
                    {
                        buffer[1] = x1;
                        for (INDEX_TYPE x2 = 0; x2 < shape[2]; ++x2)
                        {
                            if (x0 != x2 && x1 != x2)
                            {
                                buffer[2] = x2;
                                for (INDEX_TYPE x3 = 0; x3 < shape[3]; ++x3)
                                {
                                    if (x0 != x3 && x1 != x3 && x2 != x3)
                                    {
                                        buffer[3] = x3;
                                        for (INDEX_TYPE x4 = 0; x4 < shape[4];
                                             ++x4)
                                        {
                                            if (x0 != x4 && x1 != x4 &&
                                                x2 != x4 && x3 != x4)
                                            {
                                                buffer[4] = x4;
                                                for (INDEX_TYPE x5 = 0;
                                                     x5 < shape[5]; ++x5)
                                                {
                                                    if (x0 != x5 && x1 != x5 &&
                                                        x2 != x5 && x3 != x5 &&
                                                        x4 != x5)
                                                    {
                                                        buffer[5] = x5;
                                                        for (INDEX_TYPE x6 = 0;
                                                             x6 < shape[6];
                                                             ++x6)
                                                        {
                                                            if (x0 != x6 &&
                                                                x1 != x6 &&
                                                                x2 != x6 &&
                                                                x3 != x6 &&
                                                                x4 != x6 &&
                                                                x5 != x6)
                                                            {
                                                                buffer[6] = x6;
                                                                if (!f(buffer))
                                                                {
                                                                    return;
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        case 8:
        {
            for (INDEX_TYPE x0 = 0; x0 < shape[0]; ++x0)
            {
                buffer[0] = x0;
                for (INDEX_TYPE x1 = 0; x1 < shape[1]; ++x1)
                {
                    if (x0 != x1)
                    {
                        buffer[1] = x1;
                        for (INDEX_TYPE x2 = 0; x2 < shape[2]; ++x2)
                        {
                            if (x0 != x2 && x1 != x2)
                            {
                                buffer[2] = x2;
                                for (INDEX_TYPE x3 = 0; x3 < shape[3]; ++x3)
                                {
                                    if (x0 != x3 && x1 != x3 && x2 != x3)
                                    {
                                        buffer[3] = x3;
                                        for (INDEX_TYPE x4 = 0; x4 < shape[4];
                                             ++x4)
                                        {
                                            if (x0 != x4 && x1 != x4 &&
                                                x2 != x4 && x3 != x4)
                                            {
                                                buffer[4] = x4;
                                                for (INDEX_TYPE x5 = 0;
                                                     x5 < shape[5]; ++x5)
                                                {
                                                    if (x0 != x5 && x1 != x5 &&
                                                        x2 != x5 && x3 != x5 &&
                                                        x4 != x5)
                                                    {
                                                        buffer[5] = x5;
                                                        for (INDEX_TYPE x6 = 0;
                                                             x6 < shape[6];
                                                             ++x6)
                                                        {
                                                            if (x0 != x6 &&
                                                                x1 != x6 &&
                                                                x2 != x6 &&
                                                                x3 != x6 &&
                                                                x4 != x6 &&
                                                                x5 != x6)
                                                            {
                                                                buffer[6] = x6;
                                                                for (INDEX_TYPE
                                                                         x7 = 0;
                                                                     x7 <
                                                                     shape[7];
                                                                     ++x7)
                                                                {
                                                                    if (x0 !=
                                                                            x7 &&
                                                                        x1 !=
                                                                            x7 &&
                                                                        x2 !=
                                                                            x7 &&
                                                                        x3 !=
                                                                            x7 &&
                                                                        x4 !=
                                                                            x7 &&
                                                                        x5 !=
                                                                            x7 &&
                                                                        x6 !=
                                                                            x7)
                                                                    {
                                                                        buffer[7] =
                                                                            x7;
                                                                        if (!f(buffer))
                                                                        {
                                                                            return;
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        case 9:
        {
            for (INDEX_TYPE x0 = 0; x0 < shape[0]; ++x0)
            {
                buffer[0] = x0;
                for (INDEX_TYPE x1 = 0; x1 < shape[1]; ++x1)
                {
                    if (x0 != x1)
                    {
                        buffer[1] = x1;
                        for (INDEX_TYPE x2 = 0; x2 < shape[2]; ++x2)
                        {
                            if (x0 != x2 && x1 != x2)
                            {
                                buffer[2] = x2;
                                for (INDEX_TYPE x3 = 0; x3 < shape[3]; ++x3)
                                {
                                    if (x0 != x3 && x1 != x3 && x2 != x3)
                                    {
                                        buffer[3] = x3;
                                        for (INDEX_TYPE x4 = 0; x4 < shape[4];
                                             ++x4)
                                        {
                                            if (x0 != x4 && x1 != x4 &&
                                                x2 != x4 && x3 != x4)
                                            {
                                                buffer[4] = x4;
                                                for (INDEX_TYPE x5 = 0;
                                                     x5 < shape[5]; ++x5)
                                                {
                                                    if (x0 != x5 && x1 != x5 &&
                                                        x2 != x5 && x3 != x5 &&
                                                        x4 != x5)
                                                    {
                                                        buffer[5] = x5;
                                                        for (INDEX_TYPE x6 = 0;
                                                             x6 < shape[6];
                                                             ++x6)
                                                        {
                                                            if (x0 != x6 &&
                                                                x1 != x6 &&
                                                                x2 != x6 &&
                                                                x3 != x6 &&
                                                                x4 != x6 &&
                                                                x5 != x6)
                                                            {
                                                                buffer[6] = x6;
                                                                for (INDEX_TYPE
                                                                         x7 = 0;
                                                                     x7 <
                                                                     shape[7];
                                                                     ++x7)
                                                                {
                                                                    if (x0 !=
                                                                            x7 &&
                                                                        x1 !=
                                                                            x7 &&
                                                                        x2 !=
                                                                            x7 &&
                                                                        x3 !=
                                                                            x7 &&
                                                                        x4 !=
                                                                            x7 &&
                                                                        x5 !=
                                                                            x7 &&
                                                                        x6 !=
                                                                            x7)
                                                                    {
                                                                        buffer[7] =
                                                                            x7;
                                                                        for (
                                                                            INDEX_TYPE
                                                                                x8 =
                                                                                    0;
                                                                            x8 <
                                                                            shape
                                                                                [8];
                                                                            ++x8)
                                                                        {
                                                                            if (x0 !=
                                                                                    x8 &&
                                                                                x1 !=
                                                                                    x8 &&
                                                                                x2 !=
                                                                                    x8 &&
                                                                                x3 !=
                                                                                    x8 &&
                                                                                x4 !=
                                                                                    x8 &&
                                                                                x5 !=
                                                                                    x8 &&
                                                                                x6 !=
                                                                                    x8 &&
                                                                                x7 !=
                                                                                    x8)
                                                                            {
                                                                                buffer[8] =
                                                                                    x8;
                                                                                if (!f(buffer))
                                                                                {
                                                                                    return;
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        case 10:
        {
            for (INDEX_TYPE x0 = 0; x0 < shape[0]; ++x0)
            {
                buffer[0] = x0;
                for (INDEX_TYPE x1 = 0; x1 < shape[1]; ++x1)
                {
                    if (x0 != x1)
                    {
                        buffer[1] = x1;
                        for (INDEX_TYPE x2 = 0; x2 < shape[2]; ++x2)
                        {
                            if (x0 != x2 && x1 != x2)
                            {
                                buffer[2] = x2;
                                for (INDEX_TYPE x3 = 0; x3 < shape[3]; ++x3)
                                {
                                    if (x0 != x3 && x1 != x3 && x2 != x3)
                                    {
                                        buffer[3] = x3;
                                        for (INDEX_TYPE x4 = 0; x4 < shape[4];
                                             ++x4)
                                        {
                                            if (x0 != x4 && x1 != x4 &&
                                                x2 != x4 && x3 != x4)
                                            {
                                                buffer[4] = x4;
                                                for (INDEX_TYPE x5 = 0;
                                                     x5 < shape[5]; ++x5)
                                                {
                                                    if (x0 != x5 && x1 != x5 &&
                                                        x2 != x5 && x3 != x5 &&
                                                        x4 != x5)
                                                    {
                                                        buffer[5] = x5;
                                                        for (INDEX_TYPE x6 = 0;
                                                             x6 < shape[6];
                                                             ++x6)
                                                        {
                                                            if (x0 != x6 &&
                                                                x1 != x6 &&
                                                                x2 != x6 &&
                                                                x3 != x6 &&
                                                                x4 != x6 &&
                                                                x5 != x6)
                                                            {
                                                                buffer[6] = x6;
                                                                for (INDEX_TYPE
                                                                         x7 = 0;
                                                                     x7 <
                                                                     shape[7];
                                                                     ++x7)
                                                                {
                                                                    if (x0 !=
                                                                            x7 &&
                                                                        x1 !=
                                                                            x7 &&
                                                                        x2 !=
                                                                            x7 &&
                                                                        x3 !=
                                                                            x7 &&
                                                                        x4 !=
                                                                            x7 &&
                                                                        x5 !=
                                                                            x7 &&
                                                                        x6 !=
                                                                            x7)
                                                                    {
                                                                        buffer[7] =
                                                                            x7;
                                                                        for (
                                                                            INDEX_TYPE
                                                                                x8 =
                                                                                    0;
                                                                            x8 <
                                                                            shape
                                                                                [8];
                                                                            ++x8)
                                                                        {
                                                                            if (x0 !=
                                                                                    x8 &&
                                                                                x1 !=
                                                                                    x8 &&
                                                                                x2 !=
                                                                                    x8 &&
                                                                                x3 !=
                                                                                    x8 &&
                                                                                x4 !=
                                                                                    x8 &&
                                                                                x5 !=
                                                                                    x8 &&
                                                                                x6 !=
                                                                                    x8 &&
                                                                                x7 !=
                                                                                    x8)
                                                                            {
                                                                                buffer[8] =
                                                                                    x8;
                                                                                for (
                                                                                    INDEX_TYPE
                                                                                        x9 =
                                                                                            0;
                                                                                    x9 <
                                                                                    shape
                                                                                        [9];
                                                                                    ++x9)
                                                                                {
                                                                                    if (x0 !=
                                                                                            x9 &&
                                                                                        x1 !=
                                                                                            x9 &&
                                                                                        x2 !=
                                                                                            x9 &&
                                                                                        x3 !=
                                                                                            x9 &&
                                                                                        x4 !=
                                                                                            x9 &&
                                                                                        x5 !=
                                                                                            x9 &&
                                                                                        x6 !=
                                                                                            x9 &&
                                                                                        x7 !=
                                                                                            x9 &&
                                                                                        x8 !=
                                                                                            x9)
                                                                                    {
                                                                                        buffer[9] =
                                                                                            x9;
                                                                                        if (!f(buffer))
                                                                                        {
                                                                                            return;
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        default:
        {
            exitable_n_nested_loop_unique_labels_fallback(n, shape,
                                                          std::forward<F>(f));
        }
    }
}

} // namespace nxtgm
