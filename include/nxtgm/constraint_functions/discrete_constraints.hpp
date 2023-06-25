#pragma once

#include <nxtgm/nxtgm.hpp>
#include <nxtgm/constraint_functions/discrete_constraint_function_base.hpp>
namespace nxtgm{


    class PairwiseUniqueLables: public DiscreteConstraintFunctionBase {
    public:

        using DiscreteConstraintFunctionBase::feasible;

        PairwiseUniqueLables(discrete_label_type n_labels, energy_type scale = 1);

        std::size_t arity() const override;
        discrete_label_type shape(std::size_t ) const override;
        std::size_t size() const override;
        
        std::pair<bool, energy_type>  feasible(const const_discrete_label_span& discrete_labels) const override;

        void add_to_lp(IlpData & ,  const span<std::size_t> &, IlpConstraintBuilderBuffer &)const override;

     private:
        discrete_label_type n_labels_;
        energy_type scale_;
    };

}