#include  <nxtgm/nxtgm.hpp>
#include <nxtgm/spaces/discrete_space.hpp>

namespace nxtgm
{
    IndicatorVariableMapping::IndicatorVariableMapping(const DiscreteSpace & space)
    :   is_simple_(space.is_simple()),
        mapping_(space.is_simple() ? 1 : space.size())
    {
        if(is_simple_)
        {
            mapping_[0] = space[0];
            n_variables_ =  space.size() * space[0];
        }
        else
        {
            n_variables_ = 0;
            for(std::size_t vi = 0; vi <space.size(); ++vi){
                mapping_[vi] = n_variables_;
                n_variables_ += space[vi];
            }
        }
    
    }

    std::size_t IndicatorVariableMapping::operator[](std::size_t variable)const
    {
        // when simple, mapping_[0] is the number of labels
        return (is_simple_ ? variable * mapping_[0] : mapping_[variable]);
    }
}
