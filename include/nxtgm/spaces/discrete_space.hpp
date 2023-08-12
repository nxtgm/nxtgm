#pragma once
#ifndef NXTGM_DISCRETE_SPACE_HPP
#define NXTGM_DISCRETE_SPACE_HPP

#include <iostream>
#include <nlohmann/json.hpp>
#include <nxtgm/nxtgm.hpp>
#include <nxtgm/utils/n_nested_loops.hpp>
#include <vector>

namespace nxtgm
{

class DiscreteSpace
{

  private:
  public:
    nlohmann::json serialize_json() const;
    static DiscreteSpace deserialize_json(const nlohmann::json &json);

    // simple space
    inline DiscreteSpace(std::size_t n_variables, std::size_t n_labels)
        : n_labels_(1, n_labels),
          n_variables_(n_variables),
          is_simple_(true)
    {
    }

    template <class ITER>
    inline DiscreteSpace(ITER begin, ITER end)
        : n_labels_(begin, end),
          n_variables_(),
          is_simple_(false)
    {
        n_variables_ = n_labels_.size();
    }

    inline DiscreteSpace(const std::vector<discrete_label_type> &n_labels)
        : n_labels_(n_labels),
          n_variables_(n_labels.size()),
          is_simple_(false)
    {
    }

    inline nxtgm::discrete_label_type operator[](std::size_t variable) const
    {
        return is_simple_ ? n_labels_.front() : n_labels_[variable];
    }

    inline std::size_t size() const
    {
        return n_variables_;
    }

    template <class F, class SOLUTION>
    void for_each_solution(SOLUTION &solution, F &&f) const
    {
        this->exitable_for_each_solution(solution, [&](const SOLUTION &s) {
            f(s);
            return true;
        });
    }

    template <class F, class SOLUTION>
    void exitable_for_each_solution(SOLUTION &solution, F &&f) const
    {
        // since operator[] acts like the shape we can use
        // *this as shape
        exitable_n_nested_loops<discrete_label_type>(this->size(), *this, solution, std::forward<F>(f));
    }
    bool is_simple() const
    {
        return is_simple_;
    }
    discrete_label_type max_num_labels() const;

  private:
    std::vector<nxtgm::discrete_label_type> n_labels_;
    uint64_t n_variables_;
    bool is_simple_;
};

class IndicatorVariableMapping
{
  public:
    IndicatorVariableMapping(const DiscreteSpace &space);
    std::size_t operator[](std::size_t variable) const;

    inline std::size_t num_indicator_variables() const
    {
        return n_variables_;
    }

    template <class INDICATOR_VARIABLE_SOLUTION, class MODEL_SOLUTION>
    bool lp_solution_to_model_solution(const INDICATOR_VARIABLE_SOLUTION &indicator_variable_solution,
                                       MODEL_SOLUTION &model_solution)
    {
        bool all_integral = true;
        for (std::size_t vi = 0; vi < space_.size(); ++vi)
        {
            double best = 0.0;
            std::size_t best_label = 0;
            const auto mapping_begin = this->operator[](vi);
            for (discrete_label_type l = 0; l < space_[vi]; ++l)
            {
                const auto lp_sol = indicator_variable_solution[mapping_begin + l];
                if (lp_sol > best)
                {
                    best = lp_sol;
                    best_label = l;
                    if (best >= 0.99999)
                    {
                        break;
                    }
                }
            }
            if (best < 0.99999)
            {
                all_integral = false;
            }
            model_solution[vi] = best_label;
        }
        return all_integral;
    }

  private:
    const DiscreteSpace &space_;
    std::size_t n_variables_;
    std::vector<std::size_t> mapping_;
};

} // namespace nxtgm

#endif // NXTGM_DISCRETE_SPACE_HPP
