#include <iomanip>
#include <nxtgm/functions/array_constraint_function.hpp>
#include <nxtgm/functions/unique_labels_constraint_function.hpp>
#include <nxtgm/utils/n_nested_loops.hpp>

// fusion
#include <nxtgm/optimizers/gm/discrete/fusion.hpp>

#include <unordered_map>

namespace nxtgm
{

UniqueLables::UniqueLables(std::size_t arity, discrete_label_type n_labels, bool with_ignore_label,
                           discrete_label_type ignore_label, energy_type scale)
    : arity_(arity),
      n_labels_(n_labels),
      with_ignore_label_(with_ignore_label),
      ignore_label_(ignore_label),
      scale_(scale)
{
    if (arity < 2)
    {
        throw std::runtime_error("UniqueLables arity must be >= 2");
    }
}

std::size_t UniqueLables::arity() const
{
    return arity_;
}

discrete_label_type UniqueLables::num_labels() const
{
    return n_labels_;
}

energy_type UniqueLables::value(const discrete_label_type *discrete_labels) const
{
    auto n_duplicates = 0;

    for (auto i = 0; i < arity_ - 1; ++i)
    {
        const auto label_i = discrete_labels[i];
        if (with_ignore_label_ && label_i == ignore_label_)
        {
            continue;
        }
        for (auto j = i + 1; j < arity_; ++j)
        {
            const auto label_j = discrete_labels[j];
            if (with_ignore_label_ && label_j == ignore_label_)
            {
                continue;
            }
            if (label_i == label_j)
            {
                ++n_duplicates;
            }
        }
    }
    return n_duplicates == 0 ? static_cast<energy_type>(0) : scale_ * n_duplicates;
}

std::unique_ptr<DiscreteConstraintFunctionBase> UniqueLables::clone() const
{
    return std::make_unique<UniqueLables>(arity_, n_labels_, with_ignore_label_, scale_);
}

void UniqueLables::add_to_lp(IlpData &ilp_data, const std::size_t *indicator_variables_mapping) const
{
    for (discrete_label_type l = 0; l < n_labels_; ++l)
    {
        if (with_ignore_label_ && l == ignore_label_)
        {
            continue;
        }
        ilp_data.begin_constraint(0.0, 1.0);
        for (auto i = 0; i < arity_; ++i)
        {
            ilp_data.add_constraint_coefficient(indicator_variables_mapping[i] + l, 1.0);
        }
    }
}

nlohmann::json UniqueLables::serialize_json() const
{
    return {{"type", UniqueLables::serialization_key()},
            {"arity", arity_},
            {"num_labels", n_labels_},
            {"with_ignore_label", with_ignore_label_},
            {"ignore_label", ignore_label_},
            {"scale", scale_}};
}

void UniqueLables::serialize(Serializer &serializer) const
{
    serializer(UniqueLables::serialization_key());
    serializer(arity_);
    serializer(n_labels_);
    serializer(with_ignore_label_);
    serializer(ignore_label_);
    serializer(scale_);
}

std::unique_ptr<DiscreteConstraintFunctionBase> UniqueLables::deserialize(Deserializer &deserializer)
{
    auto f = new UniqueLables();
    deserializer(f->arity_);
    deserializer(f->n_labels_);
    deserializer(f->scale_);
    deserializer(f->with_ignore_label_);
    deserializer(f->ignore_label_);
    return std::unique_ptr<DiscreteConstraintFunctionBase>(f);
}

std::unique_ptr<DiscreteConstraintFunctionBase> UniqueLables::deserialize_json(const nlohmann::json &json)
{
    return std::make_unique<UniqueLables>(
        json["arity"].get<std::size_t>(), json["num_labels"].get<discrete_label_type>(),
        json["with_ignore_label"].get<bool>(), json["ignore_label"].get<discrete_label_type>(),
        json["scale"].get<energy_type>());
}

std::size_t UniqueLables::min_counts(discrete_label_type l) const
{
    return 0;
}
std::size_t UniqueLables::max_counts(discrete_label_type l) const
{
    if (with_ignore_label_ && l == ignore_label_)
    {
        return this->arity_;
    }
    return 1;
}

bool UniqueLables::with_ignore_label() const
{
    return with_ignore_label_;
}

discrete_label_type UniqueLables::ignore_label() const
{
    return ignore_label_;
}

void UniqueLables::fuse(const discrete_label_type *labels_a, const discrete_label_type *labels_b,
                        discrete_label_type *labels_ab, const std::size_t fused_arity,
                        const std::size_t *fuse_factor_var_pos, Fusion &fusion) const
{

    // return LabelCountConstraintBase::fuse(labels_a, labels_b, labels_ab, fused_arity, fuse_factor_var_pos, fusion);
    //  std::cout<<"UniqueLables::fuse:"<<std::endl;
    //  std::cout<<"fused arity: "<<fused_arity<<" / "<<arity_<<std::endl;

    xt::xtensor<energy_type, 2> fused_binary = xt::xtensor<energy_type, 2>::from_shape({2, 2});
    xt::xtensor<energy_type, 1> fused_unary = xt::xtensor<energy_type, 1>::from_shape({2});

    auto consider_label = [&](discrete_label_type l) {
        if (with_ignore_label_ && l == ignore_label_)
        {
            return false;
        }
        return true;
    };

    std::unordered_map<discrete_label_type, uint64_t> used_labels;
    for (std::size_t ai = 0; ai < arity_; ++ai)
    {
        const auto label_a = labels_a[ai];
        const auto label_b = labels_b[ai];
        if (label_a == label_b && consider_label(label_a))
        {
            used_labels[label_a] += 1;
        }
    }

    // UNARY
    for (std::size_t i = 0; i < fused_arity; ++i)
    {

        // reset the fused function data
        std::fill(fused_unary.begin(), fused_unary.end(), 0.0);
        const auto pi = fuse_factor_var_pos[i];
        const auto label_a = labels_a[pi];
        const auto label_b = labels_b[pi];

        std::size_t n_constraints = 0;
        if (consider_label(label_a))
        {
            if (auto fr = used_labels.find(label_a); fr != used_labels.end())
            {
                fused_unary(0) = used_labels[label_a];
                ++n_constraints;
            }
        }
        if (consider_label(label_b))
        {
            if (auto fr = used_labels.find(label_b); fr != used_labels.end())
            {
                fused_unary(1) = used_labels[label_b];
                ++n_constraints;
            }
        }
        if (n_constraints == 0)
        {
            // nothing todo
        }
        // else if(n_constraints == 2)
        // {

        // }
        else
        {
            // add the fused function to the fusion
            std::size_t vars[1] = {pi};
            // std::cout<<"add unary fused function"<<std::endl;
            // std::cout<<"var pos: "<<pi<<std::endl;
            fusion.add_to_fuse_gm(std::make_unique<ArrayDiscreteConstraintFunction>(fused_unary), vars);
            // std::cout<<"done"<<std::endl;
        }
    }

    // PAIRWISE
    // loop over pairs of free variables
    for (std::size_t i0 = 0; i0 < fused_arity - 1; ++i0)
    {
        const auto pi0 = fuse_factor_var_pos[i0];
        for (std::size_t i1 = i0 + 1; i1 < fused_arity; ++i1)
        {
            const auto pi1 = fuse_factor_var_pos[i1];
            // std::cout<<i0<<" / "<<i1<<std::endl;

            // reset the fused function data
            std::fill(fused_binary.begin(), fused_binary.end(), 0.0);

            // the two labels for the first variable
            const auto labels_pi0_a = labels_a[pi0];
            const auto labels_pi0_b = labels_b[pi0];

            NXTGM_CHECK_OP(labels_pi0_a, !=, labels_pi0_b, "labels_pi0_a != labels_pi0_b");

            // the two labels for the second variable
            const auto labels_pi1_a = labels_a[pi1];
            const auto labels_pi1_b = labels_b[pi1];

            NXTGM_CHECK_OP(labels_pi1_a, !=, labels_pi1_b, "labels_pi1_a != labels_pi1_b");

            // std::cout<<"labels for first  variable: "<<labels_pi0_a<<" / "<<labels_pi0_b<<std::endl;
            // std::cout<<"labels for second variable: "<<labels_pi1_a<<" / "<<labels_pi1_b<<std::endl;

            std::size_t num_constraint = 0;

            // entry 0,0
            if (consider_label(labels_pi0_a) && consider_label(labels_pi1_a) && labels_pi0_a == labels_pi1_a)
            {
                fused_binary(0, 0) = 1.0;
                ++num_constraint;
            }
            // entry 0,1
            if (consider_label(labels_pi0_a) && consider_label(labels_pi1_b) && labels_pi0_a == labels_pi1_b)
            {
                fused_binary(0, 1) = 1.0;
                ++num_constraint;
            }
            // entry 1,0
            if (consider_label(labels_pi0_b) && consider_label(labels_pi1_a) && labels_pi0_b == labels_pi1_a)
            {
                fused_binary(1, 0) = 1.0;
                ++num_constraint;
            }
            // entry 1,1
            if (consider_label(labels_pi0_b) && consider_label(labels_pi1_b) && labels_pi0_b == labels_pi1_b)
            {
                fused_binary(1, 1) = 1.0;
                ++num_constraint;
            }

            if (num_constraint == 0)
            {
                // nothing todo
            }
            // else if(num_constraint == 4)
            // {
            //     // everything is violated
            //     // do we just ignore this case?
            //     // there is nothing to do

            // }
            else
            {
                // add the fused function to the fusion
                // std::cout<<"add fused function"<<std::endl;
                std::size_t vars2[2] = {pi0, pi1};
                // std::cout<<"add binary fused function"<<std::endl;
                // std::cout<<"var pos: "<<pi0<<" / "<<pi1<<std::endl;
                fusion.add_to_fuse_gm(std::make_unique<ArrayDiscreteConstraintFunction>(fused_binary), vars2);
                // std::cout<<"done"<<std::endl;
            }
        }
    }
}

} // namespace nxtgm
