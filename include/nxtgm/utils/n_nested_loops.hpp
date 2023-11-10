#pragma once

namespace nxtgm
{

template <class INDEX_TYPE, class SHAPE, class BUFFER, class F>
void exitable_n_nested_loops_fallback(const std::size_t n, SHAPE &&shape, BUFFER &&buffer, F &&f)
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
void exitable_n_nested_loops(const std::size_t n, SHAPE &&shape, BUFFER &&buffer, F &&f)
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
        exitable_n_nested_loops_fallback<INDEX_TYPE>(n, std::forward<SHAPE>(shape), std::forward<BUFFER>(buffer),
                                                     std::forward<F>(f));
        return;
    }
}

template <class INDEX_TYPE, class SHAPE, class BUFFER, class F>
void n_nested_loops(const std::size_t n, SHAPE &&shape, BUFFER &buffer, F &&f)
{
    exitable_n_nested_loops<INDEX_TYPE>(n, std::forward<SHAPE>(shape), std::forward<BUFFER>(buffer),
                                        [&](BUFFER &buffer) {
                                            f(buffer);
                                            return true;
                                        });
}

template <class INDEX_TYPE, class SHAPE, class BUFFER, class F>
void exitable_n_nested_loop_unique_labels_fallback(const std::size_t n, SHAPE &&shape, BUFFER &&buffer, F &&f)
{
    throw std::runtime_error("Not implemented yet for n > 4");
}

template <class INDEX_TYPE, class SHAPE, class BUFFER, class F>
void exitable_n_nested_loop_unique_labels(const std::size_t n, SHAPE &&shape, BUFFER &&buffer, F &&f)
{

    switch (n)
    {
    case 1: {
        for (INDEX_TYPE x0 = 0; x0 < shape[0]; ++x0)
        {
            buffer[0] = x0;
            if (!f(buffer))
            {
                return;
            }
        }
        break;
    }
    case 2: {
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
        break;
    }
    case 3: {
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
        break;
    }
    case 4: {
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
        break;
    }
    default: {
        exitable_n_nested_loop_unique_labels_fallback<INDEX_TYPE>(n, std::forward<SHAPE>(shape),
                                                                  std::forward<BUFFER>(buffer), std::forward<F>(f));
    }
    }
}

template <class C, class F>
void n_nested_loops_binary_shape(std::size_t n, C *coordinates, F &&functor)
{
    switch (n)
    {
    case 1: {
        for (coordinates[0] = 0; coordinates[0] < 2; ++coordinates[0])
        {
            functor(coordinates);
        }
        break;
    }
    case 2: {
        for (coordinates[0] = 0; coordinates[0] < 2; ++coordinates[0])
            for (coordinates[1] = 0; coordinates[1] < 2; ++coordinates[1])
            {
                functor(coordinates);
            }
        break;
    }
    case 3: {
        for (coordinates[0] = 0; coordinates[0] < 2; ++coordinates[0])
            for (coordinates[1] = 0; coordinates[1] < 2; ++coordinates[1])
                for (coordinates[2] = 0; coordinates[2] < 2; ++coordinates[2])
                {
                    functor(coordinates);
                }
        break;
    }
    default: {
        // initialize the solution with zeros
        std::fill(coordinates, coordinates + n, 0);

        const auto last_var = n - 1;
        auto index = last_var;
        while (true)
        {
            // TODO: Your inner loop code goes here. You can inspect the values in
            // slots
            functor(coordinates);

            // Increment
            coordinates[last_var]++;
            // Carry
            while (coordinates[index] == 2)
            {
                // Overflow, we're done
                if (index == 0)
                {
                    return;
                }

                coordinates[index--] = 0;
                coordinates[index]++;
            }
            index = last_var;
        }
    }
    }
}

} // namespace nxtgm
