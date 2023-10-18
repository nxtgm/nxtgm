#include <discrete_function_tester.hpp>

#include <nxtgm/functions/label_costs_energy_function.hpp>
#include <nxtgm/functions/potts_energy_function.hpp>
#include <nxtgm/functions/sparse_energy_function.hpp>
#include <nxtgm/functions/xarray_energy_function.hpp>
#include <nxtgm/functions/xtensor_energy_function.hpp>

// this function is used to check if the default implementation of
// DiscreteEnergyFunctionBase are correct. Therefore we only implement the pure
// virtual functions and leave the rest.
namespace nxtgm
{

class DefaultTesterFunction : public DiscreteEnergyFunctionBase
{
  public:
    template <class TENSOR>
    DefaultTesterFunction(TENSOR &&values)
        : values(std::forward<TENSOR>(values))
    {
    }
    std::size_t arity() const override
    {
        return values.dimension();
    }
    std::size_t size() const override
    {
        return values.size();
    }
    discrete_label_type shape(std::size_t index) const override
    {
        return values.shape()[index];
    }

    energy_type value(const discrete_label_type *discrete_labels) const override
    {
        const_discrete_label_span l(discrete_labels, values.dimension());
        return values[l];
    }
    std::unique_ptr<DiscreteEnergyFunctionBase> clone() const override
    {
        return std::make_unique<DefaultTesterFunction>(values);
    }

    static std::unique_ptr<DiscreteEnergyFunctionBase> deserialize_json(const nlohmann::json &json)
    {
        std::vector<std::size_t> shape;
        for (auto s : json["shape"])
        {
            shape.push_back(s);
        }
        xt::xarray<energy_type> array(shape);
        std::copy(json["values"].begin(), json["values"].end(), array.begin());
        return std::make_unique<DefaultTesterFunction>(array);
    }

    inline static std::string serialization_key()
    {
        return "array";
    }

    nlohmann::json serialize_json() const override
    {
        nlohmann::json shape = nlohmann::json::array();
        for (auto s : values.shape())
        {
            shape.push_back(s);
        }

        // iterator pair to nlhohmann::json
        auto jsonvalues = nlohmann::json::array();
        for (auto it = values.begin(); it != values.end(); ++it)
        {
            jsonvalues.push_back(*it);
        }

        return {{"type", DefaultTesterFunction::serialization_key()},
                {"dimensions", values.dimension()},
                {"shape", shape},
                {"values", jsonvalues}};
    }
    void serialize(Serializer &serializer) const override
    {
        serializer(DefaultTesterFunction::serialization_key());
        serializer(values);
    }
    xt::xarray<energy_type> values;
};

TEST_CASE("xtensor")
{
    SUBCASE("1D")
    {
        XTensor<1> unary({1.0, 2.0, 3.0});
        tests::test_discrete_energy_function<XTensor<1>>(&unary);
    }
    SUBCASE("3D")
    {
        XTensor<3> function(xt::arange<int>(0, 2 * 3 * 4).reshape({2, 3, 4}));
        tests::test_discrete_energy_function<XTensor<3>>(&function);

        std::size_t vars[1] = {0};
        discrete_label_type labels[1] = {1};

        SUBCASE("bind")
        {
            auto binded = function.bind(span<const std::size_t>(vars, 1), span<const discrete_label_type>(labels, 1));

            // CHECK(binded->arity() == 2);
            // CHECK(binded->shape(0) == 3);
            // CHECK(binded->shape(1) == 4);

            // for (discrete_label_type l0 = 0; l0 < binded->shape(0); ++l0)
            // {
            //     for (discrete_label_type l1 = 0; l1 < binded->shape(0); ++l1)
            //     {
            //         CHECK(binded->value({l0, l1}) == doctest::Approx(function.value({1, l0, l1})));
            //     }
            // }
        }
    }
}

TEST_CASE("xarray")
{
    XArray function(xt::arange<int>(0, 2 * 3 * 4).reshape({2, 3, 4}));
    tests::test_discrete_energy_function<XArray>(&function);
}

TEST_CASE("potts")
{
    Potts function(3, 1.0f);
    tests::test_discrete_energy_function<Potts>(&function);
}

TEST_CASE("label-costs")
{

    SUBCASE("unittest")
    {

        LabelCosts function(3, {1.0, 2.0, 3.0, 4.0});
        CHECK(function.arity() == 3);
        CHECK(function.size() == std::pow(4, 3));
        CHECK(function.shape(0) == 4);
        CHECK(function.shape(1) == 4);
        CHECK(function.shape(2) == 4);
        CHECK(function.value({0, 0, 0}) == doctest::Approx(1.0));
        CHECK(function.value({1, 1, 1}) == doctest::Approx(2.0));
        CHECK(function.value({2, 2, 2}) == doctest::Approx(3.0));
        CHECK(function.value({3, 3, 3}) == doctest::Approx(4.0));

        CHECK(function.value({0, 1, 0}) == doctest::Approx(1.0 + 2.0));
        CHECK(function.value({0, 0, 2}) == doctest::Approx(1.0 + 3.0));
        CHECK(function.value({1, 2, 3}) == doctest::Approx(2.0 + 3.0 + 4.0));
    }

    SUBCASE("less-labels")
    {
        LabelCosts function(3, {1.0, 2});
        tests::test_discrete_energy_function<LabelCosts>(&function);
    }
    SUBCASE("same-labels")
    {
        LabelCosts function(3, {1.0, 2.0, 3.0});
        tests::test_discrete_energy_function<LabelCosts>(&function);
    }
    SUBCASE("more-labels")
    {
        LabelCosts function(3, {1.0, 2.0, 3.0, 4.0});
        tests::test_discrete_energy_function<LabelCosts>(&function);
    }

    SUBCASE("10d")
    {
        SUBCASE("less-labels")
        {
            LabelCosts function(10, {1.0, 2.0});
            tests::test_discrete_energy_function<LabelCosts>(&function);
        }
    }
}

TEST_CASE("sparse")
{

    SUBCASE("unittest")
    {
        std::vector<discrete_label_type> shape{4, 5, 6};
        SparseDiscreteEnergyFunction function(shape);
        function.data()(1, 0, 1) = 1.0;
        function.data()(3, 1, 2) = 5.0;

        CHECK(function.arity() == 3);
        CHECK(function.size() == 4 * 5 * 6);
        CHECK(function.shape(0) == 4);
        CHECK(function.shape(1) == 5);
        CHECK(function.shape(2) == 6);

        CHECK(function.value({0, 0, 0}) == doctest::Approx(0.0));
        CHECK(function.value({1, 1, 1}) == doctest::Approx(0.0));
        CHECK(function.value({2, 2, 2}) == doctest::Approx(0.0));
        CHECK(function.value({1, 2, 3}) == doctest::Approx(0.0));

        CHECK(function.value({1, 0, 1}) == doctest::Approx(1.0));
        CHECK(function.value({3, 1, 2}) == doctest::Approx(5.0));

        tests::test_discrete_energy_function<SparseDiscreteEnergyFunction>(&function);
    }
}

TEST_CASE("default-tester-function")
{
    DefaultTesterFunction function(xt::arange<int>(0, 2 * 3 * 4).reshape({2, 3, 4}));
    tests::test_discrete_energy_function<DefaultTesterFunction>(&function);
}

} // namespace nxtgm
