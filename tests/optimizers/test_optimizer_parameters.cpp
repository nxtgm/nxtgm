#include <nxtgm/optimizers/optimizer_parameters.hpp>
#include <nxtgm/utils/uany.hpp>
#include <nxtgm_test_common.hpp>
#include <vector>

TEST_CASE("optimizer-parameters")
{
    nxtgm::OptimizerParameters parameters;
    parameters["sfubar"] = "baz";
    CHECK(parameters.string_parameters["sfubar"].front() == "baz");

    parameters["ifubar"] = 42;
    CHECK(parameters.int_parameters["ifubar"].front() == 42);

    parameters["ffubar"] = 41.0;
    CHECK(parameters.double_parameters["ffubar"].front() == doctest::Approx(41.0));

    parameters.any_parameters["vector"] = std::vector<int>{1, 2, 3};

    parameters["p"] = nxtgm::OptimizerParameters();
    CHECK(parameters.optimizer_parameters["p"].front().string_parameters.empty());
    CHECK(parameters.optimizer_parameters["p"].front().int_parameters.empty());
    CHECK(parameters.optimizer_parameters["p"].front().double_parameters.empty());
    CHECK(parameters.optimizer_parameters["p"].front().optimizer_parameters.empty());

    parameters.optimizer_parameters["p"].front().string_parameters["p2"].push_back("baz2");

    // copy
    nxtgm::OptimizerParameters parameters2 = parameters;
    CHECK(parameters2.optimizer_parameters["p"].front().string_parameters["p2"].front() == "baz2");

    CHECK(parameters2.any_parameters["vector"].has_value());
    CHECK(parameters2.any_parameters["vector"].type() == typeid(std::vector<int>));

    auto &vector = nxtgm::uany_cast<std::vector<int> &>(parameters2.any_parameters["vector"]);
    CHECK(vector.size() == 3);
    CHECK(vector[0] == 1);
    CHECK(vector[1] == 2);
    CHECK(vector[2] == 3);
}
