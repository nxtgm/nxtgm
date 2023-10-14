#include <nxtgm/nxtgm.hpp>
#include <nxtgm/plugins/hocr/hocr_base.hpp>

#include <algorithm>
#include <memory>
#include <vector>

namespace nxtgm
{

std::unique_ptr<HocrBase> HocrFactoryBase::create(const DiscreteGm &gm)
{
    const auto max_arity = gm.max_arity();
    auto hocr = this->create();
    hocr->add_variables(gm.num_variables());

    std::vector<double> coeffs;
    coeffs.reserve(1 << max_arity);

    std::vector<discrete_label_type> clique_labels(max_arity);
    std::vector<std::size_t> vars(max_arity);

    for (const auto &factor : gm.factors())
    {

        const unsigned int arity = factor.arity();
        if (arity == 1)
        {
            double values[2];
            factor.copy_values(values);
            hocr->add_unary_term(values[1] - values[0], factor.variables()[0]);
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
                auto energy = factor(clique_labels.data());
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
                        vars[degree++] = factor.variables()[b];
                    }
                }

                std::sort(vars.begin(), vars.begin() + degree);
                hocr->add_term(coeffs[subset], span<const std::size_t>(vars.data(), degree));
            }
        }
    }
    return std::move(hocr);
}

} // namespace nxtgm
