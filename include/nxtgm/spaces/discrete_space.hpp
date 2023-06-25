#pragma once
#ifndef NXTGM_DISCRETE_SPACE_HPP
#define NXTGM_DISCRETE_SPACE_HPP


#include <vector>

#include <nxtgm/nxtgm.hpp>
#include <nxtgm/utils/n_nested_loops.hpp>

namespace nxtgm{

    class DiscreteSpace
    {

    public:

        // simple space
        inline DiscreteSpace(std::size_t n_variables, std::size_t n_labels)
            : n_labels_(1, n_labels), n_variables_(n_variables), is_simple_(true)
        {
        }

        template<class ITER>
        inline DiscreteSpace(ITER begin, ITER end)
            : n_labels_(begin, end), n_variables_(), is_simple_(false)
        {
            n_variables_ = n_labels_.size();
        }

        inline DiscreteSpace(const std::vector<discrete_label_type> & n_labels)
            : n_labels_(n_labels), n_variables_(n_labels.size()), is_simple_(false)
        {
        }
        
        inline nxtgm::discrete_label_type operator[](std::size_t variable)const
        {
            return is_simple_ ? n_labels_.front() : n_labels_[variable];
        }

        inline std::size_t size()const
        {
            return n_variables_;
        }


        template<class F, class SOLUTION>
        void for_each_solution(SOLUTION & solution, F && f)const{
            this->exitable_for_each_solution(solution, [&](const SOLUTION & s){
                f(s);
                return true;
            });
        }


        template<class F, class SOLUTION>
        void exitable_for_each_solution(SOLUTION & solution, F && f)const
        {
            exitable_n_nested_loops<discrete_label_type>(
                this->size(),
                [&](const auto vi)
                {
                    return this->operator[](vi);
                },
                solution,
                std::forward<F>(f)
            );
        }
        bool is_simple()const{
            return is_simple_;
        }
    private:
        std::vector<nxtgm::discrete_label_type> n_labels_;
        uint64_t n_variables_;
        bool is_simple_;
    };



    class IndicatorVariableMapping
    {   
    public:
        IndicatorVariableMapping(const DiscreteSpace & space); 
        std::size_t operator[](std::size_t variable)const;

        inline std::size_t num_indicator_variables()const
        {
            return n_variables_;
        }
    private:
        bool is_simple_;
        std::size_t n_variables_;
        std::vector<std::size_t> mapping_;
    };


    

} // namespace nxtgm::space::discrete

#endif // NXTGM_DISCRETE_SPACE_HPP