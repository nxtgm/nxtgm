#include "nxtgm_test_common.hpp"
#include <nxtgm/utils/uany.hpp>

#include <memory>
#include <string>

namespace nxtgm
{
TEST_CASE("copyable-any")
{

    std::string("hello");
    uany a = std::string("hello");

    CHECK_EQ(uany_cast<std::string>(a), std::string("hello"));
}
} // namespace nxtgm
