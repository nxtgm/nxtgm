#include <nxtgm/optimizers/gm/discrete/discrete_gm_optimizer_factory.hpp>

#include <nxtgm/optimizers/gm/discrete/belief_propagation.hpp>
#include <nxtgm/optimizers/gm/discrete/brute_force_naive.hpp>
#include <nxtgm/optimizers/gm/discrete/dynamic_programming.hpp>
#include <nxtgm/optimizers/gm/discrete/graph_cut.hpp>
#include <nxtgm/optimizers/gm/discrete/icm.hpp>
#include <nxtgm/optimizers/gm/discrete/ilp_highs.hpp>
#include <nxtgm/optimizers/gm/discrete/matching_icm.hpp>
#include <nxtgm/optimizers/gm/discrete/qpbo.hpp>

namespace nxtgm
{
std::unique_ptr<DiscreteGmOptimizerBase> discrete_gm_optimizer_factory(const DiscreteGm &gm, const std::string &name,
                                                                       const nlohmann::json parameter)
{
    if (name == "brute_force_naive")
    {
        return std::make_unique<BruteForceNaive>(gm, parameter);
    }
    else if (name == "dynamic_programming")
    {
        return std::make_unique<DynamicProgramming>(gm, parameter);
    }
    else if (name == "graph_cut")
    {
        return std::make_unique<GraphCut>(gm, parameter);
    }
    else if (name == "icm")
    {
        return std::make_unique<Icm>(gm, parameter);
    }
    else if (name == "ilp_highs")
    {
        return std::make_unique<IlpHighs>(gm, parameter);
    }
    else if (name == "matching_icm")
    {
        return std::make_unique<MatchingIcm>(gm, parameter);
    }
    else if (name == "qpbo")
    {
        return std::make_unique<Qpbo>(gm, parameter);
    }
    else if (name == "belief_propagation")
    {
        return std::make_unique<BeliefPropagation>(gm, parameter);
    }
    else
    {
        throw std::runtime_error("Unknown optimizer name: " + name);
    }
}
} // namespace nxtgm
