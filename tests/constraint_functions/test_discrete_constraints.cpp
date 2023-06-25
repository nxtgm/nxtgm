#include <test.hpp>
#include <math.h>

#include <nxtgm/constraint_functions/discrete_constraints.hpp>

TEST_CASE("test discrete constraints"){
    using label_type = std::uint8_t;
    using constraint_type =  nxtgm::constraint_functions::discrete::PairwiseUniqueLables<float, label_type>;
    auto constraint =  constraint_type(2, 10.0f);


    CHECK_EQ(constraint.arity(), 2);
    CHECK_EQ(constraint.feasible({label_type(0), label_type(0)}).second, 10.0f);
    CHECK_EQ(constraint.feasible({label_type(0), label_type(0)}).first, false);

    CHECK_EQ(constraint.feasible({label_type(0), label_type(1)}).second, 0.0f);
    CHECK_EQ(constraint.feasible({label_type(0), label_type(1)}).first, true);

}