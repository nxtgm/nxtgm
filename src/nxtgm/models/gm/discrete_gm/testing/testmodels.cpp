#include <nxtgm/models/gm/discrete_gm/testing/testmodels.hpp>

#include <sstream>

namespace nxtgm
{

DiscreteGm concat_models(const std::vector<DiscreteGm> &models)
{

    std::vector<discrete_label_type> joint_space;
    std::vector<std::size_t> offsets;

    // space and offsets and names
    std::size_t total_num_var = 0;
    auto i = 0;
    for (const auto &model : models)
    {

        const auto &space = model.space();

        offsets.push_back(total_num_var);
        total_num_var += space.size();

        for (auto vi = 0; vi < space.size(); ++vi)
        {
            joint_space.push_back(space[vi]);
        }
    }

    DiscreteGm concated_gm(joint_space.begin(), joint_space.end());

    // factors
    for (auto mi = 0; mi < models.size(); ++mi)
    {
        const auto &model = models[mi];
        const auto &offset = offsets[mi];
        for (auto fi = 0; fi < model.num_factors(); ++fi)
        {
            const auto &factor = model.factor(fi);
            std::vector<std::size_t> vars(factor.variables().begin(), factor.variables().end());
            for (auto &vi : vars)
            {
                vi += offset;
            }
            auto cloned_function = factor.function()->clone();
            auto fid = concated_gm.add_energy_function(std::move(cloned_function));
            concated_gm.add_factor(vars, fid);
        }
    }

    // constraints
    for (auto mi = 0; mi < models.size(); ++mi)
    {
        const auto &model = models[mi];
        const auto &offset = offsets[mi];
        for (auto ci = 0; ci < model.num_constraints(); ++ci)
        {
            const auto &constraint = model.constraint(ci);
            std::vector<std::size_t> vars(constraint.variables().begin(), constraint.variables().end());
            for (auto &vi : vars)
            {
                vi += offset;
            }
            auto cloned_function = constraint.function()->clone();
            auto fid = concated_gm.add_constraint_function(std::move(cloned_function));
            concated_gm.add_constraint(vars, fid);
        }
    }

    return concated_gm;
}

ConcatenatedModels::ConcatenatedModels(std::vector<std::unique_ptr<DiscreteGmTestmodel>> &&model_generators)
    : model_generators(std::move(model_generators))
{
}

ConcatenatedModels::ConcatenatedModels(std::unique_ptr<DiscreteGmTestmodel> model_gen_1,
                                       std::unique_ptr<DiscreteGmTestmodel> model_gen_2)
    : model_generators()
{
    model_generators.push_back(std::move(model_gen_1));
    model_generators.push_back(std::move(model_gen_2));
}

ConcatenatedModels::ConcatenatedModels(std::unique_ptr<DiscreteGmTestmodel> model_gen_1,
                                       std::unique_ptr<DiscreteGmTestmodel> model_gen_2,
                                       std::unique_ptr<DiscreteGmTestmodel> model_gen_3)
    : model_generators()
{
    model_generators.push_back(std::move(model_gen_1));
    model_generators.push_back(std::move(model_gen_2));
    model_generators.push_back(std::move(model_gen_3));
}

std::pair<DiscreteGm, std::string> ConcatenatedModels::operator()(unsigned seed)
{
    std::vector<DiscreteGm> models;
    std::stringstream ss;
    ss << "ConcatenatedModels(";

    for (auto &model_generator : model_generators)
    {
        auto [model, name] = model_generator->operator()(seed);
        ss << name << ", ";
        models.push_back(std::move(model));
    }
    ss << ")";

    auto concated_gm = concat_models(models);
    return std::pair<DiscreteGm, std::string>(std::move(concated_gm), ss.str());
}
std::unique_ptr<DiscreteGmTestmodel> concatenated_models(
    std::vector<std::unique_ptr<DiscreteGmTestmodel>> &&model_generators)
{
    return std::make_unique<ConcatenatedModels>(std::move(model_generators));
}

std::unique_ptr<DiscreteGmTestmodel> concatenated_models(std::unique_ptr<DiscreteGmTestmodel> model_gen_1,
                                                         std::unique_ptr<DiscreteGmTestmodel> model_gen_2)
{
    return std::make_unique<ConcatenatedModels>(std::move(model_gen_1), std::move(model_gen_2));
}
std::unique_ptr<DiscreteGmTestmodel> concatenated_models(std::unique_ptr<DiscreteGmTestmodel> model_gen_1,
                                                         std::unique_ptr<DiscreteGmTestmodel> model_gen_2,
                                                         std::unique_ptr<DiscreteGmTestmodel> model_gen_3)
{
    return std::make_unique<ConcatenatedModels>(std::move(model_gen_1), std::move(model_gen_2), std::move(model_gen_3));
}

Star::Star(std::size_t n_arms, std::size_t n_labels)
    : n_arms(n_arms),
      n_labels(n_labels)
{
}

std::pair<DiscreteGm, std::string> Star::operator()(unsigned seed)
{
    xt::random::seed(seed);

    auto space = DiscreteSpace(n_arms + 1, n_labels);
    DiscreteGm gm(space);

    // unaries
    for (std::size_t vi = 0; vi < space.size(); ++vi)
    {
        auto tensor = xt::random::rand<energy_type>({n_labels}, energy_type(-1), energy_type(1));
        auto f = std::make_unique<nxtgm::XTensor<1>>(tensor);
        auto fid = gm.add_energy_function(std::move(f));
        gm.add_factor({vi}, fid);
    }

    // pairwise
    for (std::size_t arm_index = 0; arm_index < n_arms; ++arm_index)
    {
        auto tensor = xt::random::rand<energy_type>({n_labels, n_labels}, energy_type(-1), energy_type(1));
        auto f = std::make_unique<nxtgm::XTensor<2>>(tensor);
        auto fid = gm.add_energy_function(std::move(f));
        gm.add_factor({std::size_t(0), arm_index + 1}, fid);
    }

    std::stringstream ss;
    ss << "Star(n_arms=" << n_arms << ", n_labels=" << n_labels << ", seed=" << seed << ")";
    ++seed;
    return std::pair<DiscreteGm, std::string>(std::move(gm), ss.str());
}

std::unique_ptr<DiscreteGmTestmodel> star(std::size_t n_arms, std::size_t n_labels)
{
    return std::make_unique<Star>(n_arms, n_labels);
}

PottsGrid::PottsGrid(std::size_t n_x, std::size_t n_y, discrete_label_type n_labels, bool submodular)
    : n_x(n_x),
      n_y(n_y),
      n_labels(n_labels),
      submodular(submodular)
{
}

std::pair<DiscreteGm, std::string> PottsGrid::operator()(unsigned seed)
{
    xt::random::seed(seed);

    auto n_variables = n_x * n_y;

    auto space = DiscreteSpace(n_variables, n_labels);
    DiscreteGm gm(space);

    // unary factors
    for (std::size_t vi = 0; vi < n_variables; vi++)
    {
        auto tensor = xt::random::rand<energy_type>({n_labels}, energy_type(-1), energy_type(1));
        auto f = std::make_unique<nxtgm::Unary>(tensor);
        auto fid = gm.add_energy_function(std::move(f));
        gm.add_factor({vi}, fid);
    }

    auto getvi = [&](std::size_t x, std::size_t y) -> std::size_t { return x + y * n_x; };

    // pairwise potts factors
    for (std::size_t x = 0; x < n_x; x++)
    {
        for (std::size_t y = 0; y < n_y; y++)
        {
            auto vi0 = getvi(x, y);
            if (x + 1 < n_x)
            {
                auto vi1 = getvi(x + 1, y);
                auto beta = xt::random::rand<energy_type>({1}, submodular ? 0.0 : energy_type(-1), energy_type(1))[0];
                auto f = std::make_unique<Potts>(n_labels, beta);
                auto fid = gm.add_energy_function(std::move(f));
                gm.add_factor({vi0, vi1}, fid);
            }
            if (y + 1 < n_y)
            {
                auto vi1 = getvi(x, y + 1);
                auto beta = xt::random::rand<energy_type>({1}, submodular ? 0.0 : energy_type(-1), energy_type(1))[0];
                auto f = std::make_unique<Potts>(n_labels, beta);
                auto fid = gm.add_energy_function(std::move(f));
                gm.add_factor({vi0, vi1}, fid);
            }
        }
    }

    std::stringstream ss;
    ss << "PottsGrid(n_x=" << n_x << ",n_y=" << n_y << ", n_labels=" << n_labels << ", submodular=" << submodular
       << ", seed=" << seed << ")";
    ++seed;
    return std::pair<DiscreteGm, std::string>(std::move(gm), ss.str());
}

std::unique_ptr<DiscreteGmTestmodel> potts_grid(std::size_t n_x, std::size_t n_y, discrete_label_type n_labels,
                                                bool submodular)
{
    return std::make_unique<PottsGrid>(n_x, n_y, n_labels, submodular);
}

SparsePottsChain::SparsePottsChain(std::size_t n_variables, discrete_label_type n_labels)
    : n_variables(n_variables),
      n_labels(n_labels)
{
}

std::pair<DiscreteGm, std::string> SparsePottsChain::operator()(unsigned seed)
{
    xt::random::seed(seed);

    auto space = DiscreteSpace(n_variables, n_labels);
    DiscreteGm gm(space);

    // unary factors
    for (std::size_t vi = 0; vi < n_variables; vi++)
    {
        auto tensor = xt::random::rand<energy_type>({n_labels}, energy_type(-1), energy_type(1));

        auto f = std::make_unique<nxtgm::Unary>(tensor);
        auto fid = gm.add_energy_function(std::move(f));
        gm.add_factor({vi}, fid);
    }

    // pairwise potts factors
    for (std::size_t i = 0; i < n_variables - 1; i++)
    {
        auto beta = xt::random::rand<energy_type>({1}, energy_type(-1), energy_type(1))(0);
        std::vector<std::size_t> shape = {n_labels, n_labels};
        auto f = std::make_unique<SparseDiscreteEnergyFunction>(shape);
        for (auto l = 0; l < n_labels; ++l)
        {
            f->data()(l, l) = beta;
        }
        auto fid = gm.add_energy_function(std::move(f));
        gm.add_factor({i, i + 1}, fid);
    }

    std::stringstream ss;
    ss << "SparsePottsChain(n_variables=" << n_variables << ", n_labels=" << n_labels << ", seed=" << seed << ")";
    return std::pair<DiscreteGm, std::string>(std::move(gm), ss.str());
}

std::unique_ptr<DiscreteGmTestmodel> sparse_potts_chain(std::size_t n_variables, discrete_label_type n_labels)
{
    return std::make_unique<SparsePottsChain>(n_variables, n_labels);
}

PottsChainWithLabelCosts::PottsChainWithLabelCosts(std::size_t n_variables, discrete_label_type n_labels)
    : n_variables(n_variables),
      n_labels(n_labels)
{
}
std::pair<DiscreteGm, std::string> PottsChainWithLabelCosts::operator()(unsigned seed)
{

    auto space = DiscreteSpace(n_variables, n_labels);
    DiscreteGm gm(space);

    xt::random::seed(seed);

    // unary factors
    for (std::size_t vi = 0; vi < n_variables; vi++)
    {
        auto tensor = xt::random::rand<energy_type>({n_labels}, energy_type(-1), energy_type(1));
        auto f = std::make_unique<nxtgm::Unary>(tensor);
        auto fid = gm.add_energy_function(std::move(f));
        gm.add_factor({vi}, fid);
    }

    // pairwise potts factors
    for (std::size_t i = 0; i < n_variables - 1; i++)
    {
        auto beta = xt::random::rand<energy_type>({1}, energy_type(-1), energy_type(1))(0);
        auto f = std::make_unique<Potts>(n_labels, beta);
        auto fid = gm.add_energy_function(std::move(f));
        gm.add_factor({i, i + 1}, fid);
    }

    // global label costs
    auto label_costs = xt::random::rand<energy_type>({n_labels}, energy_type(-1), energy_type(1));
    auto f = std::make_unique<LabelCosts>(n_variables, label_costs.begin(), label_costs.end());
    auto fid = gm.add_energy_function(std::move(f));
    std::vector<std::size_t> vars(n_variables);
    std::iota(std::begin(vars), std::end(vars), 0);
    gm.add_factor(vars, fid);

    std::stringstream ss;
    ss << "PottsChainWithLabelCosts(n_variables=" << n_variables << ", n_labels=" << n_labels << ", seed=" << seed
       << ")";
    ++seed;
    return std::pair<DiscreteGm, std::string>(std::move(gm), ss.str());
}

std::unique_ptr<DiscreteGmTestmodel> potts_chain_with_label_costs(std::size_t n_variables, discrete_label_type n_labels)
{
    return std::make_unique<PottsChainWithLabelCosts>(n_variables, n_labels);
}

UniqueLabelChain::UniqueLabelChain(std::size_t n_variables, discrete_label_type n_labels, bool use_pairwise_constraints)
    : n_variables(n_variables),
      n_labels(n_labels),
      use_pairwise_constraints(use_pairwise_constraints)
{
}
std::pair<DiscreteGm, std::string> UniqueLabelChain::operator()(unsigned seed)
{

    using pairwise_function_type = nxtgm::XTensor<2>;

    xt::random::seed(seed);

    auto space = DiscreteSpace(n_variables, n_labels);
    DiscreteGm gm(space);

    // unary factors
    for (std::size_t vi = 0; vi < n_variables; vi++)
    {
        auto tensor = xt::random::rand<energy_type>({n_labels}, energy_type(-1), energy_type(1));
        auto f = std::make_unique<nxtgm::Unary>(tensor);
        auto fid = gm.add_energy_function(std::move(f));
        gm.add_factor({vi}, fid);
    }

    // pairwise factors
    for (std::size_t i = 0; i < n_variables - 1; i++)
    {
        auto tensor = xt::random::rand<energy_type>({n_labels, n_labels}, energy_type(-1), energy_type(1));
        auto f = std::make_unique<nxtgm::XTensor<2>>(tensor);
        auto fid = gm.add_energy_function(std::move(f));
        gm.add_factor({i, i + 1}, fid);
    }

    // pairwise constraints (all pairs, not just neighbours)
    if (use_pairwise_constraints)
    {
        auto f = std::make_unique<UniqueLables>(2, n_labels);
        auto fid = gm.add_constraint_function(std::move(f));
        for (std::size_t i = 0; i < n_variables - 1; i++)
        {
            for (std::size_t j = i + 1; j < n_variables; j++)
            {

                gm.add_constraint({i, j}, fid);
            }
        }
    }
    else
    {
        auto f = std::make_unique<UniqueLables>(n_variables, n_labels);
        auto fid = gm.add_constraint_function(std::move(f));

        std::vector<std::size_t> vars(n_variables);
        std::iota(std::begin(vars), std::end(vars), 0);

        gm.add_constraint(vars, fid);
    }
    std::stringstream ss;
    ss << "UniqueLabelChain(n_variables=" << n_variables << ", n_labels=" << n_labels << ", seed=" << seed << ")";

    ++seed;
    return std::pair<DiscreteGm, std::string>(std::move(gm), ss.str());
}

std::unique_ptr<DiscreteGmTestmodel> unique_label_chain(std::size_t n_variables, discrete_label_type n_labels,
                                                        bool use_pairwise_constraints)
{
    return std::make_unique<UniqueLabelChain>(n_variables, n_labels, use_pairwise_constraints);
}

RandomModel::RandomModel(std::size_t n_variables, std::size_t n_factors, std::size_t max_factor_arity,
                         discrete_label_type n_labels_max)
    : n_variables(n_variables),
      n_factors(n_factors),
      max_factor_arity(max_factor_arity),
      n_labels_max(n_labels_max)
{
}
std::pair<DiscreteGm, std::string> RandomModel::operator()(unsigned seed)
{
    xt::random::seed(seed);

    // num labels
    auto num_labels = xt::eval(xt::random::randint<discrete_label_type>({n_variables}, 2, n_labels_max + 1));
    DiscreteSpace space(num_labels.begin(), num_labels.end());
    DiscreteGm gm(space);

    // all variables
    auto all_vis = xt::eval(xt::arange<std::size_t>(0, n_variables));

    for (auto fi = 0; fi < n_factors; ++fi)
    {
        xt::random::shuffle(all_vis);
        auto arity = xt::random::randint<std::size_t>({1}, 1, max_factor_arity + 1)[0];
        auto vis = xt::view(all_vis, xt::range(0, arity));
        auto shape = xt::index_view(num_labels, vis);
        auto array = xt::eval(xt::random::rand<energy_type>(shape, energy_type(-1), energy_type(1)));
        auto f = std::make_unique<nxtgm::XArray>(array);
        auto fid = gm.add_energy_function(std::move(f));
        gm.add_factor(vis, fid);
    }

    std::stringstream ss;
    ss << "RandomModel(n_variables=" << n_variables << ", n_factors=" << n_factors
       << ", max_factor_arity=" << max_factor_arity << ", n_labels_max=" << n_labels_max << ", seed=" << seed << ")";
    return std::pair<DiscreteGm, std::string>(std::move(gm), ss.str());
}

std::unique_ptr<DiscreteGmTestmodel> random_model(std::size_t n_variables, std::size_t n_factors,
                                                  std::size_t max_factor_arity, discrete_label_type n_labels_max)
{
    return std::make_unique<RandomModel>(n_variables, n_factors, max_factor_arity, n_labels_max);
}

RandomSparseModel::RandomSparseModel(std::size_t n_variables, std::size_t n_factors, std::size_t min_factor_arity,
                                     std::size_t max_factor_arity, discrete_label_type n_labels_max, float density)
    : n_variables(n_variables),
      n_factors(n_factors),
      min_factor_arity(min_factor_arity),
      max_factor_arity(max_factor_arity),
      n_labels_max(n_labels_max),
      density(density)
{
}

std::pair<DiscreteGm, std::string> RandomSparseModel::operator()(unsigned seed)
{
    xt::random::seed(seed);

    // num labels
    auto num_labels = xt::eval(xt::random::randint<discrete_label_type>({n_variables}, 2, n_labels_max + 1));
    DiscreteSpace space(num_labels.begin(), num_labels.end());
    DiscreteGm gm(space);

    // all variables
    auto all_vis = xt::eval(xt::arange<std::size_t>(0, n_variables));

    for (auto fi = 0; fi < n_factors; ++fi)
    {
        xt::random::shuffle(all_vis);
        auto arity = xt::random::randint<std::size_t>({1}, min_factor_arity, max_factor_arity + 1)[0];
        auto vis = xt::view(all_vis, xt::range(0, arity));
        auto shape = xt::index_view(num_labels, vis);

        auto f = std::make_unique<nxtgm::SparseDiscreteEnergyFunction>(shape);

        auto size = f->size();
        const auto n_nonzero = std::max(std::size_t(1), std::size_t(std::ceil(density * size)));

        for (auto i = 0; i < n_nonzero; ++i)
        {
            std::vector<discrete_label_type> coordinates(arity);
            for (auto j = 0; j < arity; ++j)
            {
                coordinates[j] = xt::random::randint<discrete_label_type>({1}, 0, shape[j])[0];
            }

            f->set_energy(coordinates.data(), xt::random::rand<energy_type>({1}, energy_type(-1), energy_type(1))[0]);
        }

        auto fid = gm.add_energy_function(std::move(f));
        gm.add_factor(vis, fid);
    }
    std::stringstream ss;
    ss << "RandomSparseModel(n_variables=" << n_variables << ", n_factors=" << n_factors
       << ", min_factor_arity=" << min_factor_arity << ", max_factor_arity=" << max_factor_arity
       << ", n_labels_max=" << n_labels_max << ", seed=" << seed << ")";
    ++seed;
    return std::pair<DiscreteGm, std::string>(std::move(gm), ss.str());
}

std::unique_ptr<DiscreteGmTestmodel> random_sparse_model(std::size_t n_variables, std::size_t n_factors,
                                                         std::size_t min_factor_arity, std::size_t max_factor_arity,
                                                         discrete_label_type n_labels_max, float density)
{
    return std::make_unique<RandomSparseModel>(n_variables, n_factors, min_factor_arity, max_factor_arity, n_labels_max,
                                               density);
}

InfeasibleModel::InfeasibleModel(std::size_t n_variables, discrete_label_type n_labels)
    : n_variables(n_variables),
      n_labels(n_labels)
{
}
std::pair<DiscreteGm, std::string> InfeasibleModel::operator()(unsigned seed)
{
    xt::random::seed(seed);

    auto space = DiscreteSpace(n_variables, n_labels);
    DiscreteGm gm(space);

    // unary factors
    for (std::size_t vi = 0; vi < n_variables; vi++)
    {
        auto tensor = xt::random::rand<energy_type>({n_labels}, energy_type(-1), energy_type(1));
        auto f = std::make_unique<nxtgm::Unary>(tensor);
        auto fid = gm.add_energy_function(std::move(f));
        gm.add_factor({vi}, fid);
    }

    // pairwise potts factors
    for (std::size_t i = 0; i < n_variables - 1; i++)
    {
        auto beta = xt::random::rand<energy_type>({1}, energy_type(-1), energy_type(1))(0);
        auto f = std::make_unique<Potts>(n_labels, beta);
    }

    // unsatisfiable constraints
    xt::xarray<energy_type> not_same = xt::zeros<energy_type>({n_labels, n_labels});
    xt::xarray<energy_type> not_different = xt::ones<energy_type>({n_labels, n_labels});
    for (std::size_t i = 0; i < n_labels; ++i)
    {
        not_same(i, i) = 1;
        not_different(i, i) = 0;
    }
    auto f_not_same = std::make_unique<ArrayDiscreteConstraintFunction>(not_same);
    auto f_not_different = std::make_unique<ArrayDiscreteConstraintFunction>(not_different);
    auto fid_not_same = gm.add_constraint_function(std::move(f_not_same));
    auto fid_not_different = gm.add_constraint_function(std::move(f_not_different));

    // pairwise constraints along the chain
    for (std::size_t i = 0; i < n_variables - 1; i++)
    {
        gm.add_constraint({i, i + 1}, fid_not_same);
        gm.add_constraint({i, i + 1}, fid_not_different);
    }

    std::stringstream ss;
    ss << "InfeasibleModel(n_variables=" << n_variables << ", n_labels=" << n_labels << ", seed=" << seed << ")";

    ++seed;
    return std::pair<DiscreteGm, std::string>(std::move(gm), ss.str());
}

std::unique_ptr<DiscreteGmTestmodel> infeasible_model(std::size_t n_variables, discrete_label_type n_labels)
{
    return std::make_unique<InfeasibleModel>(n_variables, n_labels);
}

HungarianMatchingModel::HungarianMatchingModel(std::size_t n_variables, discrete_label_type n_labels)
    : n_variables(n_variables),
      n_labels(n_labels)
{
}
std::pair<DiscreteGm, std::string> HungarianMatchingModel::operator()(unsigned seed)
{
    xt::random::seed(seed);

    auto space = DiscreteSpace(n_variables, n_labels);
    DiscreteGm gm(space);

    // unary factors
    for (std::size_t vi = 0; vi < n_variables; vi++)
    {
        auto tensor = xt::random::rand<energy_type>({n_labels}, energy_type(-1), energy_type(1));
        auto f = std::make_unique<nxtgm::Unary>(tensor);
        auto fid = gm.add_energy_function(std::move(f));
        gm.add_factor({vi}, fid);
    }

    auto f = std::make_unique<UniqueLables>(n_variables, n_labels);
    auto fid = gm.add_constraint_function(std::move(f));

    std::vector<std::size_t> vars(n_variables);
    std::iota(std::begin(vars), std::end(vars), 0);

    gm.add_constraint(vars, fid);

    std::stringstream ss;
    ss << "HungarianMatchingModel(n_variables=" << n_variables << ", n_labels=" << n_labels << ", seed=" << seed << ")";

    ++seed;
    return std::pair<DiscreteGm, std::string>(std::move(gm), ss.str());
}

std::unique_ptr<DiscreteGmTestmodel> hungarian_matching_model(std::size_t n_variables, discrete_label_type n_labels)
{
    return std::make_unique<HungarianMatchingModel>(n_variables, n_labels);
}

} // namespace nxtgm
