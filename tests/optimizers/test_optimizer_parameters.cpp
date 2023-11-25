#include <nxtgm/optimizers/optimizer_parameters.hpp>
#include <nxtgm_test_common.hpp>
#include <vector>

TEST_CASE("optimizer-parameters")
{
    nxtgm::OptimizerParameters parameters;
    parameters["sfubar"] = "baz";
    CHECK(parameters.string_parameters["sfubar"] == "baz");

    parameters["ifubar"] = 42;
    CHECK(parameters.int_parameters["ifubar"] == 42);

    parameters["ffubar"] = 41.0;
    CHECK(parameters.double_parameters["ffubar"] == doctest::Approx(41.0));

    parameters.any_parameters["vector"] = std::vector<int>{1, 2, 3};

    parameters["p"] = nxtgm::OptimizerParameters();
    CHECK(parameters.optimizer_parameters["p"].string_parameters.empty());
    CHECK(parameters.optimizer_parameters["p"].int_parameters.empty());
    CHECK(parameters.optimizer_parameters["p"].double_parameters.empty());
    CHECK(parameters.optimizer_parameters["p"].optimizer_parameters.empty());

    parameters.optimizer_parameters["p"].string_parameters["p2"] = "baz2";

    // copy
    nxtgm::OptimizerParameters parameters2 = parameters;
    CHECK(parameters2.optimizer_parameters["p"].string_parameters["p2"] == "baz2");

    CHECK(parameters2.any_parameters["vector"].has_value());
    CHECK(parameters2.any_parameters["vector"].type() == typeid(std::vector<int>));

    auto &vector = std::any_cast<std::vector<int> &>(parameters2.any_parameters["vector"]);
    CHECK(vector.size() == 3);
    CHECK(vector[0] == 1);
    CHECK(vector[1] == 2);
    CHECK(vector[2] == 3);
}
