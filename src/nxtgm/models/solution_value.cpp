#include <nxtgm/models/solution_value.hpp>

namespace nxtgm
{
    bool SolutionValue::operator<(const SolutionValue & other) const
    {
        if(is_feasible_)
        {
            if(other.is_feasible_)
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
            if(other.is_feasible_)
            {                   
                return false;
            }
            else
            {
                if(how_violated_ == other.how_violated_)
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

    bool SolutionValue::operator<=(const SolutionValue & other) const
    {
        if(is_feasible_)
        {
            if(other.is_feasible_)
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
            if(other.is_feasible_)
            {                   
                return false;
            }
            else
            {
                if(how_violated_ == other.how_violated_)
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

    // print class
    std::ostream & operator<<(std::ostream & os, const SolutionValue & solution)
    {
        os << "(" << solution.energy() << ", " << solution.is_feasible() << ", " << solution.how_violated() << ")";
        return os;
    }
}