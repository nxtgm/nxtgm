#include <nxtgm/nxtgm.hpp>
#include <nxtgm/plugins/hocr/hocr_base.hpp>

#include <algorithm>
#include <memory>
#include <vector>

namespace nxtgm
{

std::unique_ptr<HocrBase> HocrFactoryBase::create(const DiscreteGm &gm, const energy_type constraint_scaling)
{
    const auto max_arity = gm.max_arity();
    auto hocr = this->create();
    hocr->add_variables(gm.num_variables());

    std::vector<double> coeffs;
    coeffs.reserve(1 << max_arity);

    std::vector<discrete_label_type> clique_labels(max_arity);
    std::vector<std::size_t> vars(max_arity);

    gm.for_each_factor_and_constraint(
        [&](auto &&factor_or_constraint, std::size_t /*factor_or_constraint_index*/, bool is_constraint) {
            const unsigned int arity = factor_or_constraint.arity();
            if (arity == 1)
            {
                double values[2];
                factor_or_constraint.function()->copy_values(values);
                const auto value = values[1] - values[0];
                hocr->add_unary_term(is_constraint ? value * constraint_scaling : value,
                                     factor_or_constraint.variables()[0]);
            }
            else
            {
                unsigned int num_assigments = 1 << arity; // 2^arity
                // discrete_label_type clique_labels[max_arity];
                coeffs.resize(num_assigments);

                for (unsigned int subset = 1; subset < num_assigments; ++subset)
                {
                    coeffs[subset] = 0;
                }

                for (unsigned int assignment = 0; assignment < num_assigments; ++assignment)
                {
                    for (unsigned int i = 0; i < arity; ++i)
                    {
                        if (assignment & (1 << i))
                        {
                            clique_labels[i] = 1;
                        }
                        else
                        {
                            clique_labels[i] = 0;
                        }
                    }
                    const auto value = factor_or_constraint(clique_labels.data());
                    const auto energy = is_constraint ? value * constraint_scaling : value;
                    for (unsigned int subset = 1; subset < num_assigments; ++subset)
                    {
                        if (assignment & ~subset)
                        {
                            continue;
                        }
                        else
                        {
                            int parity = 0;
                            for (unsigned int b = 0; b < arity; ++b)
                            {
                                parity ^= (((assignment ^ subset) & (1 << b)) != 0);
                            }
                            coeffs[subset] += parity ? -energy : energy;
                        }
                    }
                }
                for (unsigned int subset = 1; subset < num_assigments; ++subset)
                {
                    int degree = 0;
                    for (unsigned int b = 0; b < arity; ++b)
                    {
                        if (subset & (1 << b))
                        {
                            vars[degree++] = factor_or_constraint.variables()[b];
                        }
                    }

                    std::sort(vars.begin(), vars.begin() + degree);
                    hocr->add_term(coeffs[subset], span<const std::size_t>(vars.data(), degree));
                }
            }
        });

    // for (const auto &factor : gm.factors())
    // {

    //     const unsigned int arity = factor.arity();
    //     if (arity == 1)
    //     {
    //         double values[2];
    //         factor.function()->copy_values(values);
    //         hocr->add_unary_term(values[1] - values[0], factor.variables()[0]);
    //     }
    //     else
    //     {
    //         unsigned int num_assigments = 1 << arity; // 2^arity
    //         // discrete_label_type clique_labels[max_arity];
    //         coeffs.resize(num_assigments);

    //         for (unsigned int subset = 1; subset < num_assigments; ++subset)
    //         {
    //             coeffs[subset] = 0;
    //         }

    //         for (unsigned int assignment = 0; assignment < num_assigments; ++assignment)
    //         {
    //             for (unsigned int i = 0; i < arity; ++i)
    //             {
    //                 if (assignment & (1 << i))
    //                 {
    //                     clique_labels[i] = 1;
    //                 }
    //                 else
    //                 {
    //                     clique_labels[i] = 0;
    //                 }
    //             }
    //             auto energy = factor(clique_labels.data());
    //             for (unsigned int subset = 1; subset < num_assigments; ++subset)
    //             {
    //                 if (assignment & ~subset)
    //                 {
    //                     continue;
    //                 }
    //                 else
    //                 {
    //                     int parity = 0;
    //                     for (unsigned int b = 0; b < arity; ++b)
    //                     {
    //                         parity ^= (((assignment ^ subset) & (1 << b)) != 0);
    //                     }
    //                     coeffs[subset] += parity ? -energy : energy;
    //                 }
    //             }
    //         }
    //         for (unsigned int subset = 1; subset < num_assigments; ++subset)
    //         {
    //             int degree = 0;
    //             for (unsigned int b = 0; b < arity; ++b)
    //             {
    //                 if (subset & (1 << b))
    //                 {
    //                     vars[degree++] = factor.variables()[b];
    //                 }
    //             }

    //             std::sort(vars.begin(), vars.begin() + degree);
    //             hocr->add_term(coeffs[subset], span<const std::size_t>(vars.data(), degree));
    //         }
    //     }
    // }
    return std::move(hocr);
}

} // namespace nxtgm
