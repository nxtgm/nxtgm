#ifndef NXTGM_UTILS_CRTP_BASE_HPP
#define NXTGM_UTILS_CRTP_BASE_HPP

namespace nxtgm::utils
{
    template<class DERIVED_TYPE>
    class CrtpBase{
    public:
        using derived_type = DERIVED_TYPE;
        derived_type & derived_cast() {
            return static_cast<DERIVED_TYPE &>(*this);
        }
        const derived_type & derived_cast() const {
            return static_cast<const DERIVED_TYPE &>(*this);
        }
    };
}


#endif // NXTGM_UTILS_CRTP_BASE_HPP
