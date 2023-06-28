#include <discrete_function_tester.hpp>
#include <nxtgm/energy_functions/discrete_energy_functions.hpp>

TEST_CASE("unary"){

    nxtgm::Unary unary({1.0f, 2.0f, 3.0f});
    CHECK_EQ(unary.arity(), 1);
    CHECK_EQ(unary.shape(0), 3);
    CHECK_EQ(unary.size(), 3);
    CHECK_EQ(unary.energy({0}), doctest::Approx(1.0f));
    CHECK_EQ(unary.energy({1}), doctest::Approx(2.0f));
    CHECK_EQ(unary.energy({2}), doctest::Approx(3.0f));

    
    nxtgm::tests::test_energy_function(&unary);
}


TEST_CASE("xtensor1"){

    nxtgm::XTensor<1> unary({1.0f, 2.0f, 3.0f});
    CHECK_EQ(unary.arity(), 1);
    CHECK_EQ(unary.shape(0), 3);
    CHECK_EQ(unary.size(), 3);
    CHECK_EQ(unary.energy({0}), doctest::Approx(1.0f));
    CHECK_EQ(unary.energy({1}), doctest::Approx(2.0f));
    CHECK_EQ(unary.energy({2}), doctest::Approx(3.0f));

    
    nxtgm::tests::test_energy_function(&unary);
}