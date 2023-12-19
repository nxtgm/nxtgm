#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

// we do debug checks in debug more (ie NDEBUG is not defined)
// or if NXTGM_DEBUG is defined
#ifndef NDEBUG
#define NXTGM_DEBUG
#endif

/** \def NXTGM_CHECK_OP(a,op,b,message)
    \brief macro for runtime checks

    \warning The check is done
        <B> even in Release mode </B>
        (therefore if NDEBUG <B>is</B> defined)

    \param a : first argument (like a number )
    \param op : operator (== )
    \param b : second argument (like a number )
    \param message : error message (as "my error")

    <b>Usage:</b>
    \code
        int a = 1;
        NXTGM_CHECK_OP(a, ==, 1, "this should never fail")
        NXTGM_CHECK_OP(a, >=, 2, "this should fail")
    \endcode
*/
#define NXTGM_CHECK_OP(a, op, b, message)                                                                              \
    if (!static_cast<bool>(a op b))                                                                                    \
    {                                                                                                                  \
        std::stringstream s;                                                                                           \
        s << "nxtgm Error: " << message << "\n";                                                                       \
        s << "nxtgm check :  " << #a << #op << #b << "  failed:\n";                                                    \
        s << #a " = " << a << "\n";                                                                                    \
        s << #b " = " << b << "\n";                                                                                    \
        s << "in file " << __FILE__ << ", line " << __LINE__ << "\n";                                                  \
        throw std::runtime_error(s.str());                                                                             \
    }

/** \def NXTGM_CHECK(expression,message)
    \brief macro for runtime checks

    \warning The check is done
        <B> even in Release mode </B>
        (therefore if NDEBUG <B>is</B> defined)

    \param expression : expression which can evaluate to bool
    \param message : error message (as "my error")

    <b>Usage:</b>
    \code
        int a = 1;
        NXTGM_CHECK_OP(a==1, "this should never fail")
        NXTGM_CHECK_OP(a>=2, "this should fail")
    \endcode
*/

#define NXTGM_CHECK(expression, message)                                                                               \
    if (!(expression))                                                                                                 \
    {                                                                                                                  \
        std::stringstream s;                                                                                           \
        s << message << "\n";                                                                                          \
        s << "nxtgm assertion " << #expression << " failed in file " << __FILE__ << ", line " << __LINE__              \
          << std::endl;                                                                                                \
        throw std::runtime_error(s.str());                                                                             \
    }

#define NXTGM_TEST(expression, message) NXTGM_CHECK(expression, message)

#define NXTGM_TEST_OP(a, op, b, message) NXTGM_CHECK_OP(a, op, b, message)

#define NXTGM_CHECK_EQ_TOL(a, b, tol, message)                                                                         \
    if (std::abs(a - b) > tol)                                                                                         \
    {                                                                                                                  \
        std::stringstream s;                                                                                           \
        s << message << "\n";                                                                                          \
        s << "nxtgm assertion ";                                                                                       \
        s << "\"";                                                                                                     \
        s << " | " << #a << " - " << #b << "| < " #tol << "\" ";                                                       \
        s << "  failed with:\n";                                                                                       \
        s << #a << " = " << a << "\n";                                                                                 \
        s << #b << " = " << b << "\n";                                                                                 \
        s << #tol << " = " << tol << "\n";                                                                             \
        s << "in file " << __FILE__ << ", line " << __LINE__ << "\n";                                                  \
        throw std::runtime_error(s.str());                                                                             \
    }

#define NXTGM_TEST_EQ_TOL(a, b, tol, message) NXTGM_CHECK_EQ_TOL(a, b, tol, message)

#define NXTGM_CHECK_NUMBER(number)                                                                                     \
    {                                                                                                                  \
        std::stringstream s;                                                                                           \
        s << "nxtgm assertion failed in file " << __FILE__ << ", line " << __LINE__ << std::endl;                      \
        if (std::isnan(number))                                                                                        \
            throw std::runtime_error(s.str() + " number is nan");                                                      \
        if (std::isinf(number))                                                                                        \
            throw std::runtime_error(s.str() + "number is inf");                                                       \
    }

/** \def NXTGM_ASSERT_OP(a,op,b,message)
    \brief macro for runtime checks

    \warning The check is <B>only</B> done in
        in Debug mode (therefore if NDEBUG is <B>not</B> defined)

    \param a : first argument (like a number )
    \param op : operator (== )
    \param b : second argument (like a number )
    \param message : error message (as "my error")

    <b>Usage:</b>
    \code
        int a = 1;
        NXTGM_ASSERT_OP(a, ==, 1) // will not fail here
        NXTGM_ASSERT_OP(a, >=, 2) // will fail here
    \endcode
*/
#ifndef NXTGM_DEBUG
#define NXTGM_ASSERT_OP(a, op, b)                                                                                      \
    {                                                                                                                  \
    }
#else
#define NXTGM_ASSERT_OP(a, op, b)                                                                                      \
    if (!static_cast<bool>(a op b))                                                                                    \
    {                                                                                                                  \
        std::stringstream s;                                                                                           \
        s << "nxtgm assertion :  " << #a << #op << #b << "  failed:\n";                                                \
        s << #a " = " << a << "\n";                                                                                    \
        s << #b " = " << b << "\n";                                                                                    \
        s << "in file " << __FILE__ << ", line " << __LINE__ << "\n";                                                  \
        throw std::runtime_error(s.str());                                                                             \
    }
#endif

/** \def NXTGM_ASSERT(expression,message)
    \brief macro for runtime checks

    \warning The check is <B>only</B> done in
        in Debug mode (therefore if NDEBUG is <B>not</B> defined)

    \param expression : expression which can evaluate to bool

    <b>Usage:</b>
    \code
        int a = 1;
        NXTGM_ASSERT(a == 1) // will not fail here
        NXTGM_ASSERT(a >= 2) // will fail here
    \endcode
*/

#ifndef NXTGM_DEBUG
#define NXTGM_ASSERT(expression)                                                                                       \
    {                                                                                                                  \
    }
#else
#define NXTGM_ASSERT(expression)                                                                                       \
    if (!(expression))                                                                                                 \
    {                                                                                                                  \
        std::stringstream s;                                                                                           \
        s << "nxtgm assertion " << #expression << " failed in file " << __FILE__ << ", line " << __LINE__              \
          << std::endl;                                                                                                \
        throw std::runtime_error(s.str());                                                                             \
    }
#endif

#ifdef NXTGM_DEBUG
#define NXTGM_ASSERT_EQ_TOL(a, b, tol, message) NXTGM_CHECK_EQ_TOL(a, b, tol, message)
#else
#define NXTGM_ASSERT_EQ_TOL(a, b, tol, message)
#endif
