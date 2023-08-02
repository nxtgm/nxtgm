#include <nxtgm/utils/sparse_array.hpp>
#include <test.hpp>

TEST_CASE("sparse-array")
{
    SUBCASE("basics")
    {
        nxtgm::SparseArray<int> arr({2, 3});

        CHECK_EQ(arr.size(), 2 * 3);

        CHECK_EQ(arr.shape(0), 2);
        CHECK_EQ(arr.shape(1), 3);

        CHECK_EQ(arr(0, 0), 0);
        CHECK_EQ(arr(0, 1), 0);
        CHECK_EQ(arr(0, 2), 0);
        CHECK_EQ(arr(1, 0), 0);
        CHECK_EQ(arr(1, 1), 0);
        CHECK_EQ(arr(1, 2), 0);
        CHECK_EQ(arr.num_non_zero_entries(), 0);

        arr(1, 0) = 1;
        {
            const auto &arr_const = arr;

            CHECK_EQ(arr_const(0, 0), 0);
            CHECK_EQ(arr_const(0, 1), 0);
            CHECK_EQ(arr_const(0, 2), 0);
            CHECK_EQ(arr_const(1, 0), 1);
            CHECK_EQ(arr_const(1, 1), 0);
            CHECK_EQ(arr_const(1, 2), 0);
        }
        CHECK_EQ(int(arr(0, 0)), 0);
        CHECK_EQ(int(arr(0, 1)), 0);
        CHECK_EQ(int(arr(0, 2)), 0);
        CHECK_EQ(int(arr(1, 0)), 1);
        CHECK_EQ(int(arr(1, 1)), 0);
        CHECK_EQ(int(arr(1, 2)), 0);

        CHECK_EQ(arr.num_non_zero_entries(), 1);

        std::size_t index;
        int value;
        std::tie(index, value) = *arr.non_zero_entries().begin();
        CHECK_EQ(index, 3);
        CHECK_EQ(value, 1);

        std::size_t multiindex[2];
        arr.multindex_from_flat_index(0, multiindex);
        CHECK_EQ(multiindex[0], 0);
        CHECK_EQ(multiindex[1], 0);

        arr.multindex_from_flat_index(1, multiindex);
        CHECK_EQ(multiindex[0], 0);
        CHECK_EQ(multiindex[1], 1);

        arr.multindex_from_flat_index(2, multiindex);
        CHECK_EQ(multiindex[0], 0);
        CHECK_EQ(multiindex[1], 2);

        arr.multindex_from_flat_index(3, multiindex);
        CHECK_EQ(multiindex[0], 1);
        CHECK_EQ(multiindex[1], 0);

        arr.multindex_from_flat_index(4, multiindex);
        CHECK_EQ(multiindex[0], 1);
        CHECK_EQ(multiindex[1], 1);

        arr.multindex_from_flat_index(5, multiindex);
        CHECK_EQ(multiindex[0], 1);
        CHECK_EQ(multiindex[1], 2);
    }
    SUBCASE("high-dimensional")
    {
        nxtgm::SparseArray<int> arr({2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
        CHECK_EQ(arr.size(), 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10 * 11);
        CHECK_EQ(arr.num_non_zero_entries(), 0);
        CHECK_EQ(arr.dimension(), 10);
        CHECK_EQ(arr.shape(0), 2);
        CHECK_EQ(arr.shape(1), 3);
        CHECK_EQ(arr.shape(2), 4);
        CHECK_EQ(arr.shape(3), 5);
        CHECK_EQ(arr.shape(4), 6);
        CHECK_EQ(arr.shape(5), 7);
        CHECK_EQ(arr.shape(6), 8);
        CHECK_EQ(arr.shape(7), 9);
        CHECK_EQ(arr.shape(8), 10);
        CHECK_EQ(arr.shape(9), 11);

        arr(0, 0, 0, 3, 0, 0, 4, 0, 0, 4) = 10;
        arr(1, 0, 1, 0, 2, 0, 3, 0, 0, 10) = 12;

        CHECK_EQ(arr.num_non_zero_entries(), 2);
        const auto &arr_const = arr;
        CHECK_EQ(arr_const(0, 0, 0, 3, 0, 0, 4, 0, 0, 4), 10);
        CHECK_EQ(arr_const(1, 0, 1, 0, 2, 0, 3, 0, 0, 10), 12);
    }
}
