#include <nxtgm/functions/xarray_energy_function.hpp>
#include <nxtgm/optimizers/gm/discrete/discrete_gm_optimizer_factory.hpp>
#include <nxtgm/optimizers/gm/discrete/fusion.hpp>

namespace nxtgm
{
Fusion::Fusion(const DiscreteGm &gm, OptimizerParameters &&parameters)
    : gm_(gm),
      parameters_(parameters),
      gm_to_fusegm_(gm.num_variables()),
      fusegm_to_gm_(gm.num_variables()),
      fused_factor_var_pos_(),
      fused_coords_(gm.max_arity()),
      local_sol_a_(gm.max_arity()),
      local_sol_b_(gm.max_arity()),
      local_sol_(gm.max_arity()),
      fused_gm_sol_a(gm.num_variables(), 0),
      fused_gm_sol_b(gm.num_variables(), 1)
{
    ensure_all_handled("Fusion", parameters);
}

std::size_t Fusion::build_mapping(const discrete_label_type *labels_a, const discrete_label_type *labels_b,
                                  discrete_label_type *label_fused)
{
    // count number of variables and create mapping
    std::size_t fusegm_num_var = 0;
    for (std::size_t vi_gm = 0; vi_gm < gm_.num_variables(); ++vi_gm)
    {
        if (labels_a[vi_gm] != labels_b[vi_gm])
        {
            // std::cout<<"the diff at "<<vi_gm<<" is "<<labels_a[vi_gm]<<" "<<labels_b[vi_gm]<<std::endl;
            gm_to_fusegm_[vi_gm] = fusegm_num_var;
            fusegm_to_gm_[fusegm_num_var] = vi_gm;
            ++fusegm_num_var;
        }
        else
        {
            label_fused[vi_gm] = labels_a[vi_gm];
        }
    }
    // std::cout<<"fusegm_num_var: "<<fusegm_num_var<<std::endl;
    return fusegm_num_var;
}

void Fusion::add_to_fuse_gm(std::unique_ptr<DiscreteEnergyFunctionBase> fused_function,
                            const std::size_t *local_variables)
{
    std::vector<std::size_t> fused_vi(fused_function->arity());
    auto fid = fusegm_->add_energy_function(std::move(fused_function));
    for (std::size_t i = 0; i < fused_vi.size(); ++i)
    {
        const auto pos = local_variables[i];
        const auto gm_var = gm_.factors()[current_factor_or_constraint_].variable(pos);
        fused_vi[i] = gm_to_fusegm_[gm_var];
    }
    ////std::cout<<"add f fid: "<<fid<<std::endl;
    fusegm_->add_factor(fused_vi, fid);
    ////std::cout<<"add fid end"<<std::endl;
}

void Fusion::add_to_fuse_gm(std::unique_ptr<DiscreteConstraintFunctionBase> fused_function,
                            const std::size_t *local_variables)
{
    std::vector<std::size_t> fused_vi(fused_function->arity());
    ////std::cout<<"add c fid with arity: "<<fused_function->arity()<<" and shape[0]
    ///"<<fused_function->shape(0)<<std::endl;
    auto cid = fusegm_->add_constraint_function(std::move(fused_function));
    for (std::size_t i = 0; i < fused_vi.size(); ++i)
    {
        const auto pos = local_variables[i];
        const auto gm_var = gm_.constraints()[current_factor_or_constraint_].variable(pos);
        fused_vi[i] = gm_to_fusegm_[gm_var];
    }
    fusegm_->add_constraint(fused_vi, cid);
    ////std::cout<<"add cid end"<<std::endl;
}

bool Fusion::fuse(const discrete_label_type *labels_a, // best
                  const discrete_label_type *labels_b, // proposal
                  discrete_label_type *labels_fused,   // best≠≠≠
                  SolutionValue &value_fused)
{

    // print labels A and B
    // std::cout<<"labels A ";
    for (std::size_t vi_gm = 0; vi_gm < gm_.num_variables(); ++vi_gm)
    {
        // std::cout<<labels_a[vi_gm]<<" ";
    }
    // std::cout<<std::endl;

    // std::cout<<"labels B ";
    for (std::size_t vi_gm = 0; vi_gm < gm_.num_variables(); ++vi_gm)
    {
        // std::cout<<labels_b[vi_gm]<<" ";
    }
    // std::cout<<std::endl;

    // count number of variables and create mapping
    std::size_t fusegm_num_var = build_mapping(labels_a, labels_b, labels_fused);

    // if no variables are different, return the best solution
    if (fusegm_num_var == 0)
    {
        return false;
    }

    fusegm_.reset(new DiscreteGm(fusegm_num_var, 2));

    // handle factor and constraints in the same way
    gm_.for_each_factor_and_constraint([&](auto &&factor_or_constraint, std::size_t factor_or_constraint_index,
                                           bool is_constraint) {
        current_factor_or_constraint_ = factor_or_constraint_index;
        // build the local labels
        std::size_t arity = factor_or_constraint.arity();
        fused_factor_var_pos_.clear();
        for (std::size_t vi_gm_pos = 0; vi_gm_pos < arity; ++vi_gm_pos)
        {
            const auto vi_gm = factor_or_constraint.variable(vi_gm_pos);
            local_sol_a_[vi_gm_pos] = labels_a[vi_gm];
            local_sol_b_[vi_gm_pos] = labels_b[vi_gm];

            if (labels_a[vi_gm] != labels_b[vi_gm])
            {
                fused_factor_var_pos_.push_back(vi_gm_pos);
            }
            else
            {
                local_sol_[vi_gm_pos] = labels_a[vi_gm];
            }
        }
        if (fused_factor_var_pos_.size() > 0)
        {
            factor_or_constraint.function()->fuse(local_sol_a_.data(), local_sol_b_.data(), local_sol_.data(),
                                                  fused_factor_var_pos_.size(), fused_factor_var_pos_.data(), *this);
        }
    });

    // //std::cout<<"fusegm_->num_variables(): "<<fusegm_->num_variables()<<std::endl;
    // //std::cout<<"fusegm_->num_factors(): "<<fusegm_->num_factors()<<std::endl;
    // //std::cout<<"fusegm_->num_constraints(): "<<fusegm_->num_constraints()<<std::endl;

    // evaluate the proposed solution on the fused model
    auto fval_sol_a = fusegm_->evaluate(fused_gm_sol_a.data());
    auto fval_sol_b = fusegm_->evaluate(fused_gm_sol_b.data());
    auto fused_starting_point_data = fval_sol_a < fval_sol_b ? fused_gm_sol_a.data() : fused_gm_sol_b.data();
    auto fval_best = fval_sol_a < fval_sol_b ? fval_sol_a : fval_sol_b;

    const_discrete_solution_span fused_starting_point(fused_starting_point_data, fusegm_num_var);

    // create the fusion model optimizer
    auto expected_optimizer =
        discrete_gm_optimizer_factory(*fusegm_.get(), parameters_.optimizer_name, parameters_.optimizer_parameters);
    if (!expected_optimizer)
    {
        throw std::runtime_error("Could not create optimizer: " + expected_optimizer.error());
    }
    auto optimizer = std::move(expected_optimizer.value());

    // optimize the fusion model
    optimizer->optimize(nullptr, nullptr, fused_starting_point);

    // get the solution
    auto best_fused_solution = optimizer->best_solution();

    // get the value
    // auto fvalue_fused = parameters_.numeric_stability ? fusegm_->evaluate(optimizer->best_solution()) :
    // optimizer->best_solution_value();
    auto fvalue_fused = fusegm_->evaluate(optimizer->best_solution());
    // make gm sol from fused sol
    auto make_gm_sol = [&](auto &fgm_sol, auto &gm_sol) {
        for (std::size_t vi_fused = 0; vi_fused < fusegm_num_var; ++vi_fused)
        {
            const auto vi_gm = fusegm_to_gm_[vi_fused];
            NXTGM_ASSERT_OP(vi_gm, <, gm_.num_variables());
            gm_sol[vi_gm] = fgm_sol[vi_fused] == 0 ? labels_a[vi_gm] : labels_b[vi_gm];
        }
    };

    if (fvalue_fused < fval_best)
    {

#if 1
        // std::cout<<"fgmnumvar: "<<fusegm_num_var<<std::endl;
        // std::cout<<"fval_fused: "<<fvalue_fused<<" fval_best: "<<fval_best<< " fvala "<<fval_sol_a<<" fvalb
        // "<<fval_sol_b<<std::endl;

        make_gm_sol(best_fused_solution, labels_fused);

        // print fuse gm mixed sol
        // std::cout<<"fused sol ";
        for (std::size_t fi_gm = 0; fi_gm < fusegm_num_var; ++fi_gm)
        {
            // std::cout<<best_fused_solution[fi_gm]<<" ";
        }
        // std::cout<<std::endl;

        // print fuse gm pos
        // std::cout<<"fused pos ";
        for (std::size_t fi_gm = 0; fi_gm < fusegm_num_var; ++fi_gm)
        {
            // std::cout<<fusegm_to_gm_[fi_gm]<<" ";
        }
        // std::cout<<std::endl;

        // print labels A and B
        // std::cout<<"labels A ";
        for (std::size_t vi_gm = 0; vi_gm < gm_.num_variables(); ++vi_gm)
        {
            // std::cout<<labels_a[vi_gm]<<" ";
        }
        // std::cout<<std::endl;

        // std::cout<<"labels B ";
        for (std::size_t vi_gm = 0; vi_gm < gm_.num_variables(); ++vi_gm)
        {
            // std::cout<<labels_b[vi_gm]<<" ";
        }

        // std::cout<<std::endl;
        // std::cout<<"labels F ";
        //  print labels fused
        for (std::size_t vi_gm = 0; vi_gm < gm_.num_variables(); ++vi_gm)
        {
            // std::cout<<labels_fused[vi_gm]<<" ";
        }
        // std::cout<<std::endl;

        auto actual_val_of_fused = gm_.evaluate(labels_fused);

        NXTGM_CHECK_OP(actual_val_of_fused, <, gm_.evaluate(labels_a), "");
        NXTGM_CHECK_OP(actual_val_of_fused, <, gm_.evaluate(labels_b), "");

#endif
    }

    bool changes = false;
    make_gm_sol(best_fused_solution, labels_fused);
    for (std::size_t vi_fused = 0; vi_fused < fusegm_num_var; ++vi_fused)
    {
        const auto vi_gm = fusegm_to_gm_[vi_fused];
        // const auto fsd = best_fused_solution[vi_fused] == 0 ? labels_a[vi_gm] : labels_b[vi_gm];
        if (labels_fused[vi_gm] != labels_a[vi_gm])
        {
            changes = true;
        }
    }

    if (changes)
    {
        auto new_value = gm_.evaluate(labels_fused);
        auto old_best = gm_.evaluate(labels_a);
        return new_value < old_best;

        NXTGM_CHECK_OP(new_value, <, old_best, "H");
    }

    return changes;
}

} // namespace nxtgm
