#pragma once

#include <nxtgm/nxtgm.hpp>
#include <nxtgm/energy_functions/discrete_energy_function_base.hpp>
#include <vector>
#include <xtensor/xtensor.hpp>
#include <xtensor/xarray.hpp>
// xtensor adapt
#include <xtensor/xadapt.hpp>
#include <cstdint>
#include <algorithm>

#ifndef NXTGM_NO_THREADS
#include <mutex>          // std::mutex
#endif

namespace nxtgm{

    class Potts : public DiscreteEnergyFunctionBase{
        public:
        using base_type = DiscreteEnergyFunctionBase;
        using base_type::energy;

        inline static std::string serialization_name(){
            return "potts";
        }
        
        Potts(std::size_t num_labels, energy_type beta);

        std::size_t arity() const override;
        discrete_label_type shape(std::size_t ) const override;
        std::size_t size() const override;
        energy_type energy(const discrete_label_type * discrete_labels) const override;
        std::unique_ptr<DiscreteEnergyFunctionBase> clone() const override;

        void copy_energies(energy_type * energies, discrete_label_type * ) const override;
        void add_energies(energy_type * energies, discrete_label_type * ) const override;

        static std::unique_ptr<DiscreteEnergyFunctionBase> deserialize_json(const nlohmann::json & json);
        nlohmann::json serialize_json() const override;

        private:
        std::size_t num_labels_; 
        energy_type beta_;
        
    };



    template<std::size_t ARITY>
    class XTensor : public DiscreteEnergyFunctionBase
    {
        public:
        using base_type = DiscreteEnergyFunctionBase;
        using base_type::energy;

        using xtensor_type = xt::xtensor<energy_type, ARITY>;
        inline static std::string serialization_name(){
            return "array";
        }
        XTensor(const xtensor_type & values) : 
            values_(values) 
        {
        }
        template<class TENSOR>
        XTensor(TENSOR && values) : 
            values_(std::forward<TENSOR>(values)) 
        {
        }

        discrete_label_type shape(std::size_t index) const override{
            return values_.shape()[index];   
        }

        std::size_t arity() const override {
            return ARITY;
        }   
    
        std::size_t size() const override {
            return values_.size();
        }

        energy_type energy(const discrete_label_type * discrete_labels) const override {
            const_discrete_label_span l(discrete_labels, ARITY);
            return values_[l];
        }
        std::unique_ptr<DiscreteEnergyFunctionBase> clone() const override{
            return std::make_unique<XTensor<ARITY>>(values_);
        }

        void copy_energies(energy_type * energies, discrete_label_type * ) const override{
            std::copy(values_.data(), values_.data() + values_.size(), energies);
        }
        void add_energies(energy_type * energies, discrete_label_type * ) const override{
            std::transform(values_.data(), values_.data() + values_.size(), energies, energies, std::plus<energy_type>());
        }
        static std::unique_ptr<DiscreteEnergyFunctionBase> deserialize_json(const nlohmann::json & json){
            
            auto shape = json["shape"].get<std::vector<std::size_t>>();
            auto values = json["values"].get<std::vector<energy_type>>();
            auto values_span = xt::adapt(values, shape);
            return std::make_unique<XTensor<ARITY>>(values_span);
        }
        nlohmann::json serialize_json() const override{

            nlohmann::json shape = nlohmann::json::array();
            for(auto s: values_.shape()){
                shape.push_back(s);
            }

            // iterator pair to nlhohmann::json
            auto values = nlohmann::json::array();
            for(auto it = values_.begin(); it != values_.end(); ++it){
                values.push_back(*it);
            }

            return {
                {"type", "array"},
                {"dimensions", values_.dimension()},
                {"shape", shape},
                {"values", values}
            };
        }
        
        private:
            xtensor_type values_;
    };


    using Unary = XTensor<1>;

    class XArray : public DiscreteEnergyFunctionBase
    {
        public:

        using base_type = DiscreteEnergyFunctionBase;
        using base_type::energy;

        using xarray_type = xt::xarray<energy_type>;
        inline static std::string serialization_name(){
            return "array";
        }
        template<class TENSOR>
        XArray(TENSOR && values) : 
            values_(std::forward<TENSOR>(values)) 
        {
        }

        XArray(const xarray_type & values);
        discrete_label_type shape(std::size_t index) const override;

        std::size_t arity() const override;

        std::size_t size() const override;

        energy_type energy(const discrete_label_type * discrete_labels) const override;
        std::unique_ptr<DiscreteEnergyFunctionBase> clone() const override;

        void copy_energies(energy_type * energies, discrete_label_type * ) const override;
        void add_energies(energy_type * energies, discrete_label_type * ) const override;
        static std::unique_ptr<DiscreteEnergyFunctionBase> deserialize_json(const nlohmann::json & json);
        nlohmann::json serialize_json() const override;
        private:
            xarray_type values_;
    };


    // label costs
    // pay a cost once a label is used, but only
    // once, no matter how often it is used
    class LabelCosts : public DiscreteEnergyFunctionBase
    {    
        public:

        inline static std::string serialization_name(){
            return "label-costs";
        }
        
        using base_type = DiscreteEnergyFunctionBase;
        using base_type::energy;

        inline LabelCosts(std::size_t arity, std::initializer_list<energy_type> costs) : 
            arity_(arity),
            costs_(costs),
            is_used_(costs_.size(), 0)
        {
        }
        

        template<typename ITER>
        LabelCosts(std::size_t arity, ITER begin , ITER end) : 
            arity_(arity),
            costs_(begin, end),
            is_used_(costs_.size(), 0)
        {
        }
        
        discrete_label_type shape(std::size_t index) const override;

        std::size_t arity() const override;

        std::size_t size() const override;

        energy_type energy(const discrete_label_type * discrete_labels) const override;

        std::unique_ptr<DiscreteEnergyFunctionBase> clone() const override;
        void add_to_lp(
            IlpData & ilp_data, 
            const std::size_t * indicator_variables_mapping,
            IlpFactorBuilderBuffer & buffer
        ) const override;

        static std::unique_ptr<DiscreteEnergyFunctionBase> deserialize_json(const nlohmann::json & json);
        nlohmann::json serialize_json() const override;
        private:
            std::size_t arity_;
            std::vector<energy_type> costs_;
            mutable std::vector<std::uint8_t> is_used_;

#ifndef NXTGM_NO_THREADS
            mutable std::mutex mtx_;
#endif
    };
}