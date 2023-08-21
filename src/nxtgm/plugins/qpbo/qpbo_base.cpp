#include <nxtgm/optimizers/optimizer_parameters.hpp>
#include <nxtgm/plugins/qpbo/qpbo_base.hpp>

namespace nxtgm
{

QpboBase::ProbeOptions::ProbeOptions

ProbeOptions(const OptimizerParameters &parameters)
{

    if (auto it = parameters.string_parameters.find("directed_constraints"); it != parameters.string_parameters.end())
    {
        if (it->second == "OnlyExistingEdges")
        {
            directed_constraints = DirectedConstraints::OnlyExistingEdges;
        }
        else if (it->second == "AllPossileIfEnoughSpace")
        {
            directed_constraints = DirectedConstraints::AllPossileIfEnoughSpace;
        }
        else if (it->second == "AllPossible")
        {
            directed_constraints = DirectedConstraints::AllPossible;
        }
        else
        {
            throw std::runtime_error("Invalid qpbo_directed_constraints value, must be one of: OnlyExistingEdges, "
                                     "AllPossileIfEnoughSpace, AllPossible");
        }
    }

    if (auto it = parameters.string_parameters.find("qpbo_persistencies"); it != parameters.string_parameters.end())
    {
        if (it->second == "Strong")
        {
            persistencies = Persistencies::Strong;
        }
        else if (it->second == "Weak")
        {
            persistencies = Persistencies::Weak;
        }
        else
        {
            throw std::runtime_error("Invalid qpbo_persistencies value, must be one of: Strong, Weak");
        }
    }

    if (auto it = parameters.double_parameters.find("C"); it != parameters.double_parameters.end())
    {
        C = it->second;
    }

    if (auto it = parameters.int_parameters.find("order_seed"); it != parameters.int_parameters.end())
    {
        order_seed = it->second;
    }

    if (auto it = parameters.int_parameters.find("dilation"); it != parameters.int_parameters.end())
    {
        dilation = it->second;
    }
}
} // namespace nxtgm
