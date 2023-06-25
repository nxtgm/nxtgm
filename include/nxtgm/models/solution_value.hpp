#pragma once

#include <iostream>
#include <nxtgm/nxtgm.hpp>

namespace nxtgm
{

    class SolutionValue
    {
    public:

        inline SolutionValue(
            const energy_type  energy = 0, 
            const bool is_feasible = true,
            const energy_type  how_violated = 0
        )
        :   energy_(energy),
            is_feasible_(is_feasible),
            how_violated_(how_violated)
        {

        }

        inline energy_type energy() const
        {
            return energy_;
        }

        inline bool is_feasible() const
        {
            return is_feasible_;
        }

        inline energy_type how_violated() const
        {
            return how_violated_;
        }

        
        bool operator<(const SolutionValue & other)const;
        bool operator<=(const SolutionValue & other)const;

    private:
        energy_type energy_;
        bool is_feasible_;
        energy_type how_violated_;
    };

    // print class
    std::ostream & operator<<(std::ostream & os, const SolutionValue & solution);

} 