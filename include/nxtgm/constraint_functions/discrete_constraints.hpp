#pragma once

#include <nxtgm/nxtgm.hpp>
#include <nxtgm/constraint_functions/discrete_constraint_function_base.hpp>

#include <xtensor/xarray.hpp>

namespace nxtgm{


    class PairwiseUniqueLables: public DiscreteConstraintFunctionBase {
    public:

        inline static std::string serialization_key(){
            return "pairwise-unique";
        }
        using DiscreteConstraintFunctionBase::how_violated;

        PairwiseUniqueLables(discrete_label_type n_labels, energy_type scale = 1);

        std::size_t arity() const override;
        discrete_label_type shape(std::size_t ) const override;
        std::size_t size() const override;

        energy_type how_violated(const discrete_label_type * discrete_labels) const override;
        std::unique_ptr<DiscreteConstraintFunctionBase> clone() const override;
        void add_to_lp(IlpData & ,  const std::size_t *, IlpConstraintBuilderBuffer &)const override;
        nlohmann::json serialize_json() const override;

        static std::unique_ptr<DiscreteConstraintFunctionBase> deserialize_json(const nlohmann::json & json);

     private:
        discrete_label_type n_labels_;
        energy_type scale_;
    };


    class ArrayDiscreteConstraintFunction: public DiscreteConstraintFunctionBase {
    public:

        inline static std::string serialization_key(){
            return "array";
        }

        using DiscreteConstraintFunctionBase::how_violated;

        template<class ARRAY>
        ArrayDiscreteConstraintFunction(ARRAY && how_violated)
        :   how_violated_(std::forward<ARRAY>(how_violated)){
        }

        std::size_t arity() const override;
        discrete_label_type shape(std::size_t ) const override;
        std::size_t size() const override;

        energy_type how_violated(const discrete_label_type * discrete_labels) const override;
        std::unique_ptr<DiscreteConstraintFunctionBase> clone() const override;
        void add_to_lp(IlpData & ,  const std::size_t *, IlpConstraintBuilderBuffer &)const override;
        nlohmann::json serialize_json() const override;
        static std::unique_ptr<DiscreteConstraintFunctionBase> deserialize_json(const nlohmann::json & json);

     private:

        xt::xarray<energy_type> how_violated_;

    };



}
