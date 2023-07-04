#pragma once

namespace nxtgm{


    template<
        class INDEX_TYPE,
        class SHAPE_FUNCTOR,
        class BUFFER,
        class F
    >
    void exitable_n_nested_loops_fallback(
        const std::size_t n,
        const SHAPE_FUNCTOR & shape,
        BUFFER & buffer,
        F && f
    )
    {
        // initialize the solution with zeros
        for(INDEX_TYPE i=0; i<n; ++i)
        {
            buffer[i] = 0;
        }

        const auto last_var = n - 1;
        auto index = last_var;
        while (true)
        {
            // TODO: Your inner loop code goes here. You can inspect the values in slots
            if(!f(buffer))
            {
                return;
            }

            // Increment
            buffer[last_var]++;

            // Carry
            while (buffer[index] == shape(index))
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



    template<
        class INDEX_TYPE,
        class SHAPE_FUNCTOR,
        class BUFFER,
        class F
    >
    void exitable_n_nested_loops(
        const std::size_t n,
        const SHAPE_FUNCTOR & shape,
        BUFFER & buffer,
        F && f
    )
    {
        switch(n)
        {
            case 0:
                return;
            case 1:
                for(INDEX_TYPE i = 0; i < shape(0); ++i)
                {
                    buffer[0] = i;
                    if(!f(buffer)) return;
                }
                return ;
            case 2:
                for(INDEX_TYPE i = 0; i < shape(0); ++i)
                {
                    buffer[0] = i;
                    for(INDEX_TYPE j = 0; j < shape(1); ++j)
                    {
                        buffer[1] = j;
                        if(!f(buffer)) return;
                    }
                }
                return;
            case 3:
                for(INDEX_TYPE i = 0; i < shape(0); ++i)
                {
                    buffer[0] = i;
                    for(INDEX_TYPE j = 0; j < shape(1); ++j)
                    {
                        buffer[1] = j;
                        for(INDEX_TYPE k = 0; k < shape(2); ++k)
                        {
                            buffer[2] = k;
                            if(!f(buffer)) return;
                        }
                    }
                }
                return;
            case 4:
                for(INDEX_TYPE i = 0; i < shape(0); ++i)
                {
                    buffer[0] = i;
                    for(INDEX_TYPE j = 0; j < shape(1); ++j)
                    {
                        buffer[1] = j;
                        for(INDEX_TYPE k = 0; k < shape(2); ++k)
                        {
                            buffer[2] = k;
                            for(INDEX_TYPE l = 0; l < shape(3); ++l)
                            {
                                buffer[3] = l;
                                if(!f(buffer)) return;
                            }
                        }
                    }
                }
                return;
            default:
                exitable_n_nested_loops_fallback<INDEX_TYPE>(n, shape, buffer, std::forward<F>(f));
                return;
        }
    }

    template<
        class INDEX_TYPE,
        class SHAPE_FUNCTOR,
        class BUFFER,
        class F
    >
    void n_nested_loops(
        const std::size_t n,
        const SHAPE_FUNCTOR & shape,
        BUFFER & buffer,
        F && f
    )
    {
        exitable_n_nested_loops<INDEX_TYPE>(
            n,
            shape,
            buffer,
            [&](BUFFER & buffer){
                f(buffer);
                return true;
            }
        );
    }
}
