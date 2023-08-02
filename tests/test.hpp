#pragma once
#ifndef NXTGM_TESTS_TEST_HPP
#define NXTGM_TESTS_TEST_HPP
#define DOCTEST_CONFIG_ASSERTS_RETURN_VALUES

#include <doctest/doctest.h>

#ifdef _WIN32
#define SKIP_WIN doctest::skip(true)
#else
#define SKIP_WIN doctest::skip(false)
#endif

#endif // NXTGM_TESTS_TEST_HPP
