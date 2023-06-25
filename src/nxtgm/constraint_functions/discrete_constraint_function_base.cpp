#include <nxtgm/constraint_functions/discrete_constraint_function_base.hpp>

namespace nxtgm{


    void IlpConstraintBuilderBuffer::ensure_size(std::size_t max_constraint_size, std::size_t max_constraint_arity)
    {
        if(feasible_buffer.size() < max_constraint_size){
            feasible_buffer.resize(max_constraint_size*2);
        }
        if(label_buffer.size() < max_constraint_arity){
            label_buffer.resize(max_constraint_arity*2);
        }
        if(shape_buffer.size() < max_constraint_arity){
            shape_buffer.resize(max_constraint_arity*2);
        }
    }


        std::size_t DiscreteConstraintFunctionBase::size() const {
            std::size_t result = 1;
            for(std::size_t i = 0; i < arity(); ++i){
                result *= shape(i);
            }
            return result;
        }
        std::pair<bool, energy_type> DiscreteConstraintFunctionBase::feasible(std::initializer_list<discrete_label_type> labels) const {
            return this->feasible(const_discrete_label_span(labels.begin(), labels.size()));
        }
        void  DiscreteConstraintFunctionBase::add_to_lp(
            IlpData & ilp_data,  const span<std::size_t> & indicator_variables_mapping, IlpConstraintBuilderBuffer & buffer
        )const{
            throw std::runtime_error("Not implemented");
        }


}