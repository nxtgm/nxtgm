#include <test.hpp>

#include <fmt/core.h>
#include <nxtgm/version.hpp>

TEST_CASE("testing the factorial function")
{
    CHECK_EQ(NXTGM_VERSION,
             fmt::format("{}.{}.{}", NXTGM_VERSION_MAJOR, NXTGM_VERSION_MINOR,
                         NXTGM_VERSION_PATCH));
}
