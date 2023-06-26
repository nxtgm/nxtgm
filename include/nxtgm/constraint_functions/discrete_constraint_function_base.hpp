#pragma once


#include <nxtgm/nxtgm.hpp>
#include <nxtgm/utils/lp.hpp>

namespace nxtgm{



    struct IlpConstraintBuilderBuffer
    {   
        void ensure_size(std::size_t max_constraint_size, std::size_t max_constraint_arity);
        std::vector<uint8_t>               feasible_buffer;
        std::vector<discrete_label_type>   label_buffer;
        std::vector<discrete_label_type>   shape_buffer;
    };

    class DiscreteConstraintFunctionBase {
        public:
  
        virtual ~DiscreteConstraintFunctionBase() = default;
        
        virtual std::size_t arity() const = 0;
        virtual discrete_label_type shape(std::size_t index) const = 0;

        virtual std::pair<bool, energy_type>  feasible(const const_discrete_label_span& discrete_labels) const  = 0;

        // convenience function 
        virtual std::size_t size() const;
        virtual std::pair<bool, energy_type> feasible(std::initializer_list<discrete_label_type> labels) const;
        virtual std::unique_ptr<DiscreteConstraintFunctionBase> clone() const = 0;

        virtual void add_to_lp(
            IlpData & ilp_data,  const span<std::size_t> & indicator_variables_mapping, IlpConstraintBuilderBuffer & buffer
        ) const;

    };

}