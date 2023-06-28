#include <nxtgm/energy_functions/discrete_energy_functions.hpp>

#include <cmath>

namespace nxtgm{


    Unary::Unary(const std::vector<energy_type>& values) : 
        values_(values) 
    {

    }
    std::size_t Unary::arity() const  {
        return 1;
    }

    discrete_label_type Unary::shape(std::size_t ) const  {
        return static_cast<discrete_label_type>(values_.size());
    }

    energy_type Unary::energy(const discrete_label_type * discrete_labels) const  {
        return values_[discrete_labels[0]];
    }

    std::unique_ptr<DiscreteEnergyFunctionBase> Unary::clone() const{
        return std::make_unique<Unary>(values_);
    }
  

    Potts::Potts(std::size_t num_labels, energy_type beta) : 
        num_labels_(num_labels),
        beta_(beta)
    {
    }

    std::size_t Potts::arity() const  {
        return 2;
    }

    discrete_label_type Potts::shape(std::size_t ) const  {
        return num_labels_;
    }   
    
    std::size_t Potts::size() const  {
        return num_labels_ * num_labels_;
    }

    energy_type Potts::energy(const discrete_label_type * discrete_labels) const  {
        return beta_ * (discrete_labels[0] != discrete_labels[1]);
    }
    std::unique_ptr<DiscreteEnergyFunctionBase> Potts::clone() const{
        return std::make_unique<Potts>(num_labels_, beta_);
    }

    Xarray::Xarray(const xarray_type & values) : 
    values_(values) 
    {
    }

    discrete_label_type Xarray::shape(std::size_t index) const {
        return values_.shape()[index];   
    }

    std::size_t Xarray::arity() const  {
        return values_.dimension();
    }   

    std::size_t Xarray::size() const  {
        return values_.size();
    }

    energy_type Xarray::energy(const discrete_label_type * discrete_labels) const  {
        const_discrete_label_span discrete_labels_span(discrete_labels, values_.dimension());
        return values_[discrete_labels_span];
    }
    std::unique_ptr<DiscreteEnergyFunctionBase> Xarray::clone() const{
        return std::make_unique<Xarray>(values_);
    }

    discrete_label_type LabelCosts::shape(std::size_t index) const {
        return costs_.size();
    }

    std::size_t LabelCosts::arity() const {
        return arity_;
    }

    std::size_t LabelCosts::size() const {
        return std::pow(costs_.size(), arity_);
    }

    energy_type LabelCosts::energy(const discrete_label_type * discrete_labels) const {
        
        #ifndef NXTGM_NO_THREADS
        std::lock_guard<std::mutex> lck (mtx_);
        #endif

        std::fill(is_used_.begin(), is_used_.end(), 0);
        for(std::size_t i = 0; i < arity_; ++i){
            is_used_[discrete_labels[i]] = 1;
        }
        energy_type result = 0;
        for(std::size_t i = 0; i < is_used_.size(); ++i){
            result += is_used_[i] * costs_[i];
        }
        return result;
    }   

    std::unique_ptr<DiscreteEnergyFunctionBase> LabelCosts::clone() const{
        return std::make_unique<LabelCosts>(arity_, costs_.begin(), costs_.end());
    }


    void LabelCosts::add_to_lp(
        IlpData & ilp_data, 
        const std::size_t * indicator_variables_mapping,
        IlpFactorBuilderBuffer & buffer
    ) const
    {
        const auto label_indicator_variables_begin = ilp_data.num_variables();

        // add n_labels varialbes
        ilp_data.add_variables(0,1, costs_.begin(), costs_.end(), false);

        for(std::size_t ai=0; ai<arity_; ++ai){


            for(discrete_label_type l=0; l<static_cast<discrete_label_type>(costs_.size()); ++l){

                ilp_data.begin_constraint(0.0, 1.0);
                ilp_data.add_constraint_coefficient(label_indicator_variables_begin +l, 1.0);
                ilp_data.add_constraint_coefficient(indicator_variables_mapping[ai] + l, -1.0);
            }
        }


        for(discrete_label_type l=0; l<static_cast<discrete_label_type>(costs_.size()); ++l){
            ilp_data.begin_constraint(-1.0*arity_, 0);
            ilp_data.add_constraint_coefficient(
                label_indicator_variables_begin + l , 
                1.0);
            
            for(std::size_t ai=0; ai<arity_; ++ai){
                ilp_data.add_constraint_coefficient(
                    indicator_variables_mapping[ai] + l, 
                    -1.0);
            }
        }

    }



}