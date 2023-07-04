#pragma once

#include <nxtgm/models/gm/discrete_gm.hpp>
#include <nxtgm/optimizers/optimizer_base.hpp>

namespace nxtgm
{

    class DiscreteGmOptimizerBase : public OptimizerBase<DiscreteGm, DiscreteGmOptimizerBase>{
        public:
            using base_type = OptimizerBase<DiscreteGm, DiscreteGmOptimizerBase>;
            using base_type::base_type;
            using base_type::optimize;
    };
}
