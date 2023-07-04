
#include <nxtgm/constraint_functions/discrete_constraints.hpp>

#include <nxtgm/utils/n_nested_loops.hpp>

namespace nxtgm
{
    
    PairwiseUniqueLables::PairwiseUniqueLables(discrete_label_type n_labels, energy_type scale) 
    :   n_labels_(n_labels),
        scale_(scale){
    }

    std::size_t PairwiseUniqueLables::arity() const  {
        return 2;
    }

    discrete_label_type PairwiseUniqueLables::shape(std::size_t ) const  {
        return n_labels_;
    }   
    
    std::size_t PairwiseUniqueLables::size() const  {
        return n_labels_ * n_labels_;
    }

    
    energy_type  PairwiseUniqueLables::how_violated(const discrete_label_type * discrete_labels) const {
        return discrete_labels[0] != discrete_labels[1] ? static_cast<energy_type>(0) :scale_;
    }

    std::unique_ptr<DiscreteConstraintFunctionBase> PairwiseUniqueLables::clone() const {
        return std::make_unique<PairwiseUniqueLables>(n_labels_, scale_);
    }

    void PairwiseUniqueLables::add_to_lp(
        IlpData & ilp_data,  
        const std::size_t * indicator_variables_mapping, 
        IlpConstraintBuilderBuffer & /*buffer*/ 
    )const
    {
        for(discrete_label_type l=0; l < n_labels_; ++l){

            ilp_data.begin_constraint(0.0, 1.0);
            ilp_data.add_constraint_coefficient(indicator_variables_mapping[0] + l, 1.0);
            ilp_data.add_constraint_coefficient(indicator_variables_mapping[1] + l, 1.0);

        }
        
    }


    nlohmann::json PairwiseUniqueLables::serialize_json() const {
        return {
            {"type", PairwiseUniqueLables::serialization_key()},
            {"num_labels", n_labels_},
            {"scale", scale_}
        };
    }

    std::unique_ptr<DiscreteConstraintFunctionBase> PairwiseUniqueLables::deserialize_json(const nlohmann::json & json)
    {
        return std::make_unique<PairwiseUniqueLables>(
            json["num_labels"].get<discrete_label_type>(),
            json["scale"].get<energy_type>()
        );
    }

    

    std::size_t ArrayDiscreteConstraintFunction::arity() const{
        return how_violated_.dimension();
    }
    discrete_label_type ArrayDiscreteConstraintFunction::shape(std::size_t i) const{
        return how_violated_.shape(i);
    }
    std::size_t ArrayDiscreteConstraintFunction::size() const{
        return how_violated_.size();
    }

    energy_type ArrayDiscreteConstraintFunction::how_violated(const discrete_label_type * discrete_labels) const{
        const_discrete_label_span labels(discrete_labels, how_violated_.dimension());
        return how_violated_[labels];
    }
    std::unique_ptr<DiscreteConstraintFunctionBase> ArrayDiscreteConstraintFunction::clone() const{
        return std::make_unique<ArrayDiscreteConstraintFunction>(how_violated_);
    }
    void ArrayDiscreteConstraintFunction::add_to_lp(IlpData & ilp_data,  const std::size_t * indicator_variables_mapping, IlpConstraintBuilderBuffer & buffer)const
    {
        const auto arity = this->arity();
        auto shapef = [this](std::size_t index) { return this->shape(index); };

        if(buffer.label_buffer.size() < arity){
            buffer.label_buffer.resize(arity*2);
        }

        auto flat_index = 0;
        n_nested_loops<discrete_label_type>(arity, shapef, buffer.label_buffer, [&](auto && _){
            auto hv = how_violated_[flat_index];
            if(hv > constraint_feasiblility_limit)
            {
                ilp_data.begin_constraint(0.0, arity - 1);
                for(std::size_t i = 0; i < arity; ++i){
                    ilp_data.add_constraint_coefficient(indicator_variables_mapping[i] + buffer.label_buffer[i], 1.0);
                }
            }
            ++flat_index;
        });
    }

    nlohmann::json ArrayDiscreteConstraintFunction::serialize_json() const {
        nlohmann::json shape = nlohmann::json::array();
        for(auto s: how_violated_.shape()){
            shape.push_back(s);
        }

        // iterator pair to nlhohmann::json
        auto values = nlohmann::json::array();
        for(auto it = how_violated_.begin(); it != how_violated_.end(); ++it){
            values.push_back(*it);
        }

        return {
            {"type", "array"},
            {"dimensions", how_violated_.dimension()},
            {"shape", shape},
            {"values", values}
        };
    }


    std::unique_ptr<DiscreteConstraintFunctionBase> ArrayDiscreteConstraintFunction::deserialize_json(const nlohmann::json & json)
    {
        std::vector<std::size_t> shape;
        for(auto s: json["shape"]){
            shape.push_back(s);
        }
        xt::xarray<energy_type> array(shape);
        std::copy(json["values"].begin(), json["values"].end(), array.begin());

        return std::make_unique<ArrayDiscreteConstraintFunction>(array);
    }
} 
