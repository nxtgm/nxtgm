#include <discrete_function_tester.hpp>
#include <nxtgm/energy_functions/discrete_energy_functions.hpp>


// this function is used to check if the default implementation of DiscreteEnergyFunctionBase
// are correct. Therefore we only implement the pure virtual functions and leave the rest.

class DefaultTesterFunction : public nxtgm::DiscreteEnergyFunctionBase{
public:
    template<class TENSOR>
    DefaultTesterFunction(TENSOR && values) : 
        values(std::forward<TENSOR>(values)) 
    {
    }
    std::size_t arity() const override{
        return values.dimension();
    }
    std::size_t size() const override{
        return values.size();
    }
    nxtgm::discrete_label_type shape(std::size_t index) const override{
        return values.shape()[index];
    }

    nxtgm::energy_type energy(const  nxtgm::discrete_label_type * discrete_labels) const override{
        nxtgm::const_discrete_label_span l(discrete_labels, values.dimension());
        return values[l];
    }
    std::unique_ptr<DiscreteEnergyFunctionBase> clone() const override{
        return std::make_unique<DefaultTesterFunction>(values);
    }

   static std::unique_ptr<DiscreteEnergyFunctionBase> deserialize_json(const nlohmann::json & json){
        std::vector<std::size_t> shape;
        for(auto s: json["shape"]){
            shape.push_back(s);
        }
        xt::xarray< nxtgm::energy_type> array(shape);
        std::copy(json["values"].begin(), json["values"].end(), array.begin());
        return std::make_unique<DefaultTesterFunction>(array);
    }


    inline static std::string serialization_name(){
        return "array";
    }

    nlohmann::json serialize_json() const override{
        nlohmann::json shape = nlohmann::json::array();
        for(auto s: values.shape()){
            shape.push_back(s);
        }

        // iterator pair to nlhohmann::json
        auto jsonvalues = nlohmann::json::array();
        for(auto it = values.begin(); it != values.end(); ++it){
            jsonvalues.push_back(*it);
        }

        return {
            {"type",  DefaultTesterFunction::serialization_name()},
            {"dimensions", values.dimension()},
            {"shape", shape},
            {"values", jsonvalues}
        };
    }
    xt::xarray< nxtgm::energy_type> values;
};


TEST_CASE("xtensor"){
    SUBCASE("1D"){
        nxtgm::XTensor<1> unary({1.0, 2.0, 3.0});
        nxtgm::tests::test_discrete_energy_function<nxtgm::XTensor<1>>(&unary);
    }
    SUBCASE("3D"){
        nxtgm::XTensor<3> function( xt::arange<int>(0, 2*3*4).reshape({2,3,4}));
        nxtgm::tests::test_discrete_energy_function<nxtgm::XTensor<3>>(&function);
    }
}

TEST_CASE("xarray"){
    nxtgm::XArray function( xt::arange<int>(0, 2*3*4).reshape({2,3,4}));
    nxtgm::tests::test_discrete_energy_function<nxtgm::XArray>(&function);
}


TEST_CASE("potts"){
    nxtgm::Potts function(3, 1.0f);
    nxtgm::tests::test_discrete_energy_function<nxtgm::Potts>(&function);
}

TEST_CASE("label-costs"){

    SUBCASE("unittest"){

        nxtgm::LabelCosts function(3, {1.0, 2.0, 3.0, 4.0});
        CHECK(function.arity() == 3);
        CHECK(function.size() == std::pow(4, 3));
        CHECK(function.shape(0) == 4);
        CHECK(function.shape(1) == 4);
        CHECK(function.shape(2) == 4);
        CHECK(function.energy({0,0,0}) == doctest::Approx(1.0));
        CHECK(function.energy({1,1,1}) == doctest::Approx(2.0));
        CHECK(function.energy({2,2,2}) == doctest::Approx(3.0));
        CHECK(function.energy({3,3,3}) == doctest::Approx(4.0));

        CHECK(function.energy({0,1,0}) == doctest::Approx(1.0 + 2.0));
        CHECK(function.energy({0,0,2}) == doctest::Approx(1.0 + 3.0));
        CHECK(function.energy({1,2,3}) == doctest::Approx(2.0 + 3.0 + 4.0));
    }
    
    SUBCASE("less-labels")
    {
        nxtgm::LabelCosts function(3, {1.0, 2});
        nxtgm::tests::test_discrete_energy_function<nxtgm::LabelCosts>(&function);
    }
    SUBCASE("same-labels")
    {
        nxtgm::LabelCosts function(3, {1.0, 2.0, 3.0});
        nxtgm::tests::test_discrete_energy_function<nxtgm::LabelCosts>(&function);
    }
    SUBCASE("more-labels")
    {
        nxtgm::LabelCosts function(3, {1.0, 2.0, 3.0, 4.0});
        nxtgm::tests::test_discrete_energy_function<nxtgm::LabelCosts>(&function);
    }

    SUBCASE("10d")
    {
        SUBCASE("less-labels")
        {
            nxtgm::LabelCosts function(10, {1.0, 2.0});
            nxtgm::tests::test_discrete_energy_function<nxtgm::LabelCosts>(&function);
        }
    }
}


TEST_CASE("default-tester-function"){
    DefaultTesterFunction function( xt::arange<int>(0, 2*3*4).reshape({2,3,4}));
    nxtgm::tests::test_discrete_energy_function<DefaultTesterFunction>(&function);
}

