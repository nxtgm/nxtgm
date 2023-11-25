#pragma once
#ifndef NXTGM_TESTS_TEST_HPP
#define NXTGM_TESTS_TEST_HPP
#define DOCTEST_CONFIG_ASSERTS_RETURN_VALUES

#include <doctest/doctest.h>

// json
#include <nlohmann/json.hpp>
using njson = nlohmann::json;

#ifdef _WIN32
#define SKIP_WIN doctest::skip(true)
#else
#define SKIP_WIN doctest::skip(false)
#endif

namespace nxtgm
{
std::vector<std::string> all_optimizers();
std::vector<std::string> all_ilp_plugins();

} // namespace nxtgm

#endif // NXTGM_TESTS_TEST_HPP
