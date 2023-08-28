#include <nxtgm/models/solution_value.hpp>

namespace nxtgm
{
bool SolutionValue::operator<(const SolutionValue &other) const
{
    if (this->is_feasible())
    {
        if (other.is_feasible())
        {
            return energy_ < other.energy_;
        }
        else
        {
            return true;
        }
    }
    else
    {
        if (other.is_feasible())
        {
            return false;
        }
        else
        {
            if (how_violated_ == other.how_violated_)
            {
                return energy_ < other.energy_;
            }
            else
            {
                return how_violated_ < other.how_violated_;
            }
        }
    }
}

bool SolutionValue::operator<=(const SolutionValue &other) const
{
    if (this->is_feasible())
    {
        if (other.is_feasible())
        {
            return energy_ <= other.energy_;
        }
        else
        {
            return true;
        }
    }
    else
    {
        if (other.is_feasible())
        {
            return false;
        }
        else
        {
            if (how_violated_ == other.how_violated_)
            {
                return energy_ <= other.energy_;
            }
            else
            {
                return how_violated_ <= other.how_violated_;
            }
        }
    }
}

SolutionValue operator+(const SolutionValue &lhs, const SolutionValue &rhs)
{
    return SolutionValue(lhs.energy() + rhs.energy(), lhs.how_violated() + rhs.how_violated());
}

SolutionValue &SolutionValue::operator+=(const SolutionValue &other)
{
    energy_ += other.energy_;
    how_violated_ += other.how_violated_;
    return *this;
}
SolutionValue &SolutionValue::operator-=(const SolutionValue &other)
{
    energy_ -= other.energy_;
    how_violated_ -= other.how_violated_;
    return *this;
}

// print class
std::ostream &operator<<(std::ostream &os, const SolutionValue &solution)
{
    os << "(" << solution.energy() << ", " << solution.how_violated() << ")";
    return os;
}
} // namespace nxtgm
