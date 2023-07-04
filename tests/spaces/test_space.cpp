#include <test.hpp>

#include <vector>

#include <nxtgm/nxtgm.hpp>
#include <nxtgm/spaces/discrete_space.hpp>

#include <sstream>

TEST_CASE("discrete-space") {
  using solution_type = std::vector<nxtgm::discrete_label_type>;

  const std::size_t num_vars = 3;
  nxtgm::DiscreteSpace space(std::vector<nxtgm::discrete_label_type>{
      nxtgm::discrete_label_type(2), nxtgm::discrete_label_type(3),
      nxtgm::discrete_label_type(4)});

  auto solution = solution_type(space.size());

  std::vector<solution_type> solutions;
  std::vector<solution_type> solutions_shoulds;

  for (auto l0 = 0; l0 < space[0]; ++l0)
    for (auto l1 = 0; l1 < space[1]; ++l1)
      for (auto l2 = 0; l2 < space[2]; ++l2) {
        solution_type sol(space.size());
        sol[0] = l0;
        sol[1] = l1;
        sol[2] = l2;
        solutions_shoulds.push_back(sol);
      }

  space.for_each_solution(solution, [&](const solution_type &solution) {
    solutions.push_back(solution);
  });

  CHECK_EQ(solutions.size(), solutions_shoulds.size());
  for (std::size_t i = 0; i < solutions.size(); ++i) {
    CHECK_EQ(solutions[i], solutions_shoulds[i]);
  }
}

TEST_CASE("discrete-space-serialization") {

  SUBCASE("non-simple") {
    const std::size_t num_vars = 3;
    nxtgm::DiscreteSpace space(std::vector<nxtgm::discrete_label_type>{
        nxtgm::discrete_label_type(2), nxtgm::discrete_label_type(3),
        nxtgm::discrete_label_type(4)});

    auto as_json = space.serialize_json();

    auto jspace = nxtgm::DiscreteSpace::deserialize_json(as_json);

    CHECK_EQ(space.is_simple(), jspace.is_simple());
    CHECK_EQ(space.size(), jspace.size());
    for (std::size_t i = 0; i < space.size(); ++i) {
      CHECK_EQ(space[i], jspace[i]);
    }
  }
  SUBCASE("simple") {
    nxtgm::DiscreteSpace space(10, 3);

    auto as_json = space.serialize_json();
    auto jspace = nxtgm::DiscreteSpace::deserialize_json(as_json);

    CHECK_EQ(space.is_simple(), jspace.is_simple());
    CHECK_EQ(space.size(), jspace.size());
    for (std::size_t i = 0; i < space.size(); ++i) {
      CHECK_EQ(space[i], jspace[i]);
    }
  }
}
