#pragma once
#include <nxtgm/nxtgm.hpp>
// #include <nxtgm/models/gm/discrete_gm/gm.hpp> // this leads to a circular dependency
#include <nxtgm/models/solution_value.hpp>
#include <nxtgm/optimizers/optimizer_parameters.hpp>

namespace nxtgm
{
// forward declaration
class DiscreteGm;
class DiscreteEnergyFunctionBase;
class DiscreteConstraintFunctionBase;

class parameters_type
{
  public:
    inline parameters_type(OptimizerParameters &parameters)
    {
        parameters.assign_and_pop("optimizer_name", optimizer_name);
        parameters.assign_and_pop("optimizer_parameters", optimizer_parameters);
        parameters.assign_and_pop("numeric_stability", numeric_stability);
    }

    std::string optimizer_name = "icm";
    OptimizerParameters optimizer_parameters;
    bool numeric_stability = true;
};

class Fusion
{

  public:
    Fusion(const DiscreteGm &gm, OptimizerParameters &&parameters);

    bool fuse(const discrete_label_type *labels_a, const discrete_label_type *labels_b,
              discrete_label_type *labels_fused, SolutionValue &value_fused);

    void add_to_fuse_gm(std::unique_ptr<DiscreteEnergyFunctionBase> fused_function, const std::size_t *variables);
    void add_to_fuse_gm(std::unique_ptr<DiscreteConstraintFunctionBase> fused_function, const std::size_t *variables);

  private:
    std::size_t build_mapping(const discrete_label_type *labels_a, const discrete_label_type *labels_b,
                              discrete_label_type *label_fuseds);

    std::size_t build_factor_mapping(const discrete_label_type *labels_a, const discrete_label_type *labels_b,
                                     const std::vector<std::size_t> &factor_vi);

    const DiscreteGm &gm_;
    std::unique_ptr<DiscreteGm> fusegm_;
    parameters_type parameters_;
    std::vector<std::size_t> gm_to_fusegm_;
    std::vector<std::size_t> fusegm_to_gm_;

    std::vector<std::size_t> fused_factor_var_pos_;

    std::vector<std::size_t> fused_coords_;
    std::vector<discrete_label_type> local_sol_a_;
    std::vector<discrete_label_type> local_sol_b_;
    std::vector<discrete_label_type> local_sol_;

    std::vector<discrete_label_type> fused_gm_sol_a;
    std::vector<discrete_label_type> fused_gm_sol_b;

    std::size_t current_factor_or_constraint_ = 0;
};

} // namespace nxtgm
