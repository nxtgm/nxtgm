#pragma once


#include <nxtgm/nxtgm.hpp>
#include <nxtgm/utils/lp.hpp>
#include <nlohmann/json.hpp>
namespace nxtgm{



    struct IlpConstraintBuilderBuffer
    {   
        void ensure_size(std::size_t max_constraint_size, std::size_t max_constraint_arity);
        std::vector<energy_type>           how_violated_buffer;
        std::vector<discrete_label_type>   label_buffer;
        std::vector<discrete_label_type>   shape_buffer;
    };

    class DiscreteConstraintFunctionBase {
        public:
  
        virtual ~DiscreteConstraintFunctionBase() = default;
        
        virtual std::size_t arity() const = 0;
        virtual discrete_label_type shape(std::size_t index) const = 0;

        virtual energy_type  how_violated(const discrete_label_type * discrete_labels) const  = 0;

        // convenience function 
        virtual std::size_t size() const;
        virtual energy_type how_violated(std::initializer_list<discrete_label_type> labels) const;
        virtual std::unique_ptr<DiscreteConstraintFunctionBase> clone() const = 0;

        virtual void add_to_lp(
            IlpData & ilp_data,  const std::size_t * indicator_variables_mapping, IlpConstraintBuilderBuffer & buffer
        ) const;
        virtual nlohmann::json serialize_json() const = 0;
    };

}