#pragma once
#include <cstdint>
#include <vector>

namespace nxtgm {
class IlpData {
public:
  using value_tye = double;

  inline int add_variable(value_tye lb, value_tye ub, value_tye obj,
                          bool is_integer) {
    col_lower_.push_back(lb);
    col_upper_.push_back(ub);
    col_cost_.push_back(obj);
    is_integer_.push_back(is_integer);
    return col_lower_.size() - 1;
  }
  inline int add_variables(std::size_t n, value_tye lb, value_tye ub,
                           value_tye obj, bool is_integer) {
    for (std::size_t i = 0; i < n; ++i) {
      col_lower_.push_back(lb);
      col_upper_.push_back(ub);
      col_cost_.push_back(obj);
      is_integer_.push_back(is_integer);
    }
    return col_lower_.size() - 1;
  }
  template <class ITER>
  inline int add_variables(value_tye lb, value_tye ub, ITER obj_begin,
                           ITER obj_end, bool is_integer) {
    while (obj_begin != obj_end) {
      col_lower_.push_back(lb);
      col_upper_.push_back(ub);
      col_cost_.push_back(*obj_begin);
      is_integer_.push_back(is_integer);
      ++obj_begin;
    }
    return col_lower_.size() - 1;
  }

  template <class COEFFICIENTS, class VARS>
  inline int add_constraint(value_tye lb, value_tye ub,
                            COEFFICIENTS &&coefficients, VARS &&variables) {
    row_lower_.push_back(lb);
    row_upper_.push_back(ub);
    astart_.push_back(aindex_.size());

    const auto n_coefficients = coefficients.size();
    for (std::size_t i = 0; i < n_coefficients; ++i) {
      aindex_.push_back(variables[i]);
      avalue_.push_back(coefficients[i]);
    }
    return row_lower_.size() - 1;
  }

  inline void add_objective(std::size_t var, value_tye obj) {
    col_cost_[var] += obj;
  }

  inline void begin_constraint(const value_tye lb, const value_tye ub) {
    row_lower_.push_back(lb);
    row_upper_.push_back(ub);
    astart_.push_back(aindex_.size());
  }

  inline void add_constraint_coefficient(const std::size_t var,
                                         const value_tye coeff) {
    aindex_.push_back(var);
    avalue_.push_back(coeff);
  }

  auto &col_cost() { return col_cost_; }
  auto &col_lower() { return col_lower_; }
  auto &col_upper() { return col_upper_; }
  auto &row_lower() { return row_lower_; }
  auto &row_upper() { return row_upper_; }
  auto &astart() { return astart_; }
  auto &aindex() { return aindex_; }
  auto &avalue() { return avalue_; }
  auto &is_integer() { return is_integer_; }

  value_tye &operator[](std::size_t i) { return col_cost_[i]; }
  const value_tye &operator[](std::size_t i) const { return col_cost_[i]; }

  std::size_t num_variables() const { return col_cost_.size(); }

private:
  std::vector<value_tye> col_cost_;
  std::vector<value_tye> col_lower_;
  std::vector<value_tye> col_upper_;
  std::vector<value_tye> row_lower_;
  std::vector<value_tye> row_upper_;
  std::vector<int> astart_;
  std::vector<int> aindex_;
  std::vector<value_tye> avalue_;
  std::vector<std::uint8_t> is_integer_;
};

} // namespace nxtgm
