#pragma once

#include <nxtgm/nxtgm.hpp>

namespace nxtgm
{

class SolutionValue
{
  public:
    inline SolutionValue(const energy_type energy = 0, const energy_type how_violated = 0)
        : energy_(energy),
          how_violated_(how_violated)
    {
    }

    inline energy_type energy() const
    {
        return energy_;
    }
    inline energy_type &energy()
    {
        return energy_;
    }

    inline bool is_feasible() const
    {
        return how_violated_ < constraint_feasiblility_limit;
    }

    inline energy_type how_violated() const
    {
        return how_violated_;
    }
    inline energy_type &how_violated()
    {
        return how_violated_;
    }

    bool operator>(const SolutionValue &other) const;
    bool operator<(const SolutionValue &other) const;
    bool operator<=(const SolutionValue &other) const;
    SolutionValue &operator+=(const SolutionValue &other);
    SolutionValue &operator-=(const SolutionValue &other);

  private:
    energy_type energy_;
    energy_type how_violated_;
};

// free operator
SolutionValue operator+(const SolutionValue &lhs, const SolutionValue &rhs);

} // namespace nxtgm
