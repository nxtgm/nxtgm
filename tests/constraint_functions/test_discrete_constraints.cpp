#include <discrete_function_tester.hpp>
#include <math.h>
#include <nxtgm/constraint_functions/discrete_constraints.hpp>
#include <test.hpp>

TEST_CASE("unique-labels")
{
    SUBCASE("2")
    {
        auto constraint = nxtgm::UniqueLables(2, 2, 10.0);
        CHECK_EQ(constraint.arity(), 2);
        CHECK_EQ(constraint.how_violated({0, 0}), doctest::Approx(10.0));
        CHECK_EQ(constraint.how_violated({0, 1}), doctest::Approx(0.0));
        CHECK_EQ(constraint.how_violated({1, 1}), doctest::Approx(10.0));
        nxtgm::tests::test_discrete_constraint_function<nxtgm::UniqueLables>(
            &constraint);
    }
    SUBCASE("4")
    {
        auto constraint = nxtgm::UniqueLables(2, 4, 10.0);
        CHECK_EQ(constraint.arity(), 2);
        CHECK_EQ(constraint.how_violated({0, 0}), doctest::Approx(10.0));
        CHECK_EQ(constraint.how_violated({0, 2}), doctest::Approx(0.0));
        CHECK_EQ(constraint.how_violated({3, 3}), doctest::Approx(10.0));
        nxtgm::tests::test_discrete_constraint_function<nxtgm::UniqueLables>(
            &constraint);
    }
}

TEST_CASE("array-constraint")
{
    SUBCASE("1D")
    {
        xt::xarray<nxtgm::energy_type> how_violated = {0.0, 0.0, 1.0};
        nxtgm::ArrayDiscreteConstraintFunction constraint(how_violated);
        CHECK_EQ(constraint.arity(), 1);
        CHECK_EQ(constraint.how_violated({0}), doctest::Approx(0.0));
        CHECK_EQ(constraint.how_violated({1}), doctest::Approx(0.0));
        CHECK_EQ(constraint.how_violated({2}), doctest::Approx(1.0));
        nxtgm::tests::test_discrete_constraint_function<
            nxtgm::ArrayDiscreteConstraintFunction>(&constraint);
    }
    SUBCASE("2D")
    {
        xt::xarray<nxtgm::energy_type> how_violated =
            xt::zeros<nxtgm::energy_type>({3, 3});
        how_violated(2, 1) = 1.0;
        nxtgm::ArrayDiscreteConstraintFunction constraint(how_violated);
        CHECK_EQ(constraint.arity(), 2);
        CHECK_EQ(constraint.how_violated({0, 0}), doctest::Approx(0.0));
        CHECK_EQ(constraint.how_violated({0, 1}), doctest::Approx(0.0));
        CHECK_EQ(constraint.how_violated({0, 2}), doctest::Approx(0.0));
        CHECK_EQ(constraint.how_violated({1, 0}), doctest::Approx(0.0));
        CHECK_EQ(constraint.how_violated({1, 1}), doctest::Approx(0.0));
        CHECK_EQ(constraint.how_violated({1, 2}), doctest::Approx(0.0));
        CHECK_EQ(constraint.how_violated({2, 0}), doctest::Approx(0.0));
        CHECK_EQ(constraint.how_violated({2, 1}), doctest::Approx(1.0));
        CHECK_EQ(constraint.how_violated({2, 2}), doctest::Approx(0.0));
        nxtgm::tests::test_discrete_constraint_function<
            nxtgm::ArrayDiscreteConstraintFunction>(&constraint);
    }
}
