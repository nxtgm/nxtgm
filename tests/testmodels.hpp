#pragma once

#include <nxtgm/constraint_functions/discrete_constraints.hpp>
#include <nxtgm/energy_functions/discrete_energy_functions.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <nxtgm/models/gm/discrete_gm.hpp>
#include <nxtgm/spaces/discrete_space.hpp>
#include <random>
#include <vector>

#include <fmt/core.h>

namespace nxtgm::tests {

struct Star {
  std::pair<DiscreteGm, std::string> operator()() {
    xt::random::seed(seed);

    auto space = DiscreteSpace(n_arms + 1, n_labels);
    DiscreteGm gm(space);

    // unaries
    for (std::size_t vi = 0; vi < space.size(); ++vi) {
      auto tensor = xt::random::rand<energy_type>({n_labels}, energy_type(-1),
                                                  energy_type(1));
      auto f = std::make_unique<nxtgm::XTensor<1>>(tensor);
      auto fid = gm.add_energy_function(std::move(f));
      gm.add_factor({vi}, fid);
    }

    // pairwise
    for (std::size_t arm_index = 0; arm_index < n_arms; ++arm_index) {
      auto tensor = xt::random::rand<energy_type>(
          {n_labels, n_labels}, energy_type(-1), energy_type(1));
      auto f = std::make_unique<nxtgm::XTensor<2>>(tensor);
      auto fid = gm.add_energy_function(std::move(f));
      gm.add_factor({std::size_t(0), arm_index + 1}, fid);
    }
    const std::string name = fmt::format(
        "Star(n_arms={}, n_labels={}, seed={})", n_arms, n_labels, seed);
    return std::pair<DiscreteGm, std::string>(std::move(gm), "Star");
  }

  std::size_t n_arms = 3;
  std::size_t n_labels = 2;
  std::uint32_t seed = 0;
};

struct PottsChain {

  std::pair<DiscreteGm, std::string> operator()() {
    xt::random::seed(seed);

    auto space = DiscreteSpace(n_variables, n_labels);
    DiscreteGm gm(space);

    // unary factors
    for (std::size_t vi = 0; vi < n_variables; vi++) {
      auto tensor = xt::random::rand<energy_type>({n_labels}, energy_type(-1),
                                                  energy_type(1));
      auto f = std::make_unique<nxtgm::Unary>(tensor);
      auto fid = gm.add_energy_function(std::move(f));
      gm.add_factor({vi}, fid);
    }

    // pairwise potts factors
    for (std::size_t i = 0; i < n_variables - 1; i++) {
      auto beta = xt::random::rand<energy_type>({1}, energy_type(-1),
                                                energy_type(1))(0);
      auto f = std::make_unique<Potts>(n_labels, beta);
    }

    const std::string name =
        fmt::format("PottsChain(n_variables={}, n_labels={}, seed={})",
                    n_variables, n_labels, seed);
    ++seed;
    return std::pair<DiscreteGm, std::string>(std::move(gm), name);
  }

  std::size_t n_variables;
  discrete_label_type n_labels;
  std::uint32_t seed = 0;
};

struct PottsChainWithLabelCosts {
  std::pair<DiscreteGm, std::string> operator()() {

    auto space = DiscreteSpace(n_variables, n_labels);
    DiscreteGm gm(space);

    xt::random::seed(seed);

    // unary factors
    for (std::size_t vi = 0; vi < n_variables; vi++) {
      auto tensor = xt::random::rand<energy_type>({n_labels}, energy_type(-1),
                                                  energy_type(1));
      auto f = std::make_unique<nxtgm::Unary>(tensor);
      auto fid = gm.add_energy_function(std::move(f));
      gm.add_factor({vi}, fid);
    }

    // pairwise potts factors
    for (std::size_t i = 0; i < n_variables - 1; i++) {
      auto beta = xt::random::rand<energy_type>({1}, energy_type(-1),
                                                energy_type(1))(0);
      auto f = std::make_unique<Potts>(n_labels, beta);
    }

    // global label costs
    auto label_costs = xt::random::rand<energy_type>(
        {n_labels}, energy_type(-1), energy_type(1));
    auto f = std::make_unique<LabelCosts>(n_variables, label_costs.begin(),
                                          label_costs.end());
    auto fid = gm.add_energy_function(std::move(f));
    std::vector<std::size_t> vars(n_variables);
    std::iota(std::begin(vars), std::end(vars), 0);
    gm.add_factor(vars, fid);

    const std::string name = fmt::format(
        "PottsChainWithLabelCosts(n_variables={}, n_labels={}, seed={})",
        n_variables, n_labels, seed);
    ++seed;
    return std::pair<DiscreteGm, std::string>(std::move(gm), name);
  }

  std::size_t n_variables;
  discrete_label_type n_labels;
  std::uint32_t seed = 0;
};

struct UniqueLabelChain {

  std::pair<DiscreteGm, std::string> operator()() {

    using pairwise_function_type = nxtgm::XTensor<2>;

    xt::random::seed(seed);

    auto space = DiscreteSpace(n_variables, n_labels);
    DiscreteGm gm(space);

    // unary factors
    for (std::size_t vi = 0; vi < n_variables; vi++) {
      auto tensor = xt::random::rand<energy_type>({n_labels}, energy_type(-1),
                                                  energy_type(1));
      auto f = std::make_unique<nxtgm::Unary>(tensor);
      auto fid = gm.add_energy_function(std::move(f));
      gm.add_factor({vi}, fid);
    }

    // pairwise factors
    for (std::size_t i = 0; i < n_variables - 1; i++) {
      auto tensor = xt::random::rand<energy_type>(
          {n_labels, n_labels}, energy_type(-1), energy_type(1));
      auto f = std::make_unique<nxtgm::XTensor<2>>(tensor);
      auto fid = gm.add_energy_function(std::move(f));
      gm.add_factor({i, i + 1}, fid);
    }

    // pairwise constraints (all pairs, not just neighbours)
    auto f = std::make_unique<PairwiseUniqueLables>(n_labels);
    auto fid = gm.add_constraint_function(std::move(f));
    for (std::size_t i = 0; i < n_variables - 1; i++) {
      for (std::size_t j = i + 1; j < n_variables; j++) {

        gm.add_constraint({i, j}, fid);
      }
    }
    const std::string name =
        fmt::format("UniqueLabelChain(n_variables={}, n_labels={}, seed={})",
                    n_variables, n_labels, seed);
    ++seed;
    return std::pair<DiscreteGm, std::string>(std::move(gm), name);
  }

  std::size_t n_variables = 3;
  discrete_label_type n_labels = 3;
  std::uint32_t seed = 0;
};

struct RandomModel {

  std::pair<DiscreteGm, std::string> operator()() {
    xt::random::seed(seed);

    // num labels
    auto num_labels = xt::eval(xt::random::randint<discrete_label_type>(
        {n_variables}, 2, n_labels_max + 1));
    DiscreteSpace space(num_labels.begin(), num_labels.end());
    DiscreteGm gm(space);

    // all variables
    auto all_vis = xt::eval(xt::arange<std::size_t>(0, n_variables));

    for (auto fi = 0; fi < n_factors; ++fi) {
      xt::random::shuffle(all_vis);
      auto arity =
          xt::random::randint<std::size_t>({1}, 1, max_factor_arity + 1)[0];
      auto vis = xt::view(all_vis, xt::range(0, arity));
      auto shape = xt::index_view(num_labels, vis);
      auto array = xt::eval(xt::random::rand<energy_type>(
          shape, energy_type(-1), energy_type(1)));
      auto f = std::make_unique<nxtgm::XArray>(array);
      auto fid = gm.add_energy_function(std::move(f));
      gm.add_factor(vis, fid);
    }

    const std::string name = fmt::format(
        "RandomModel(n_variables={}, n_factors{}, max_factor_arity={}, "
        "n_labels_max={}, seed={})",
        n_variables, n_factors, max_factor_arity, n_labels_max, seed);
    ++seed;
    return std::pair<DiscreteGm, std::string>(std::move(gm), name);
  }

  std::size_t n_variables = 3;
  std::size_t n_factors = 4;
  std::size_t max_factor_arity = 3;
  discrete_label_type n_labels_max = 3;
  std::uint32_t seed = 0;
};

struct InfeasibleModel {

  std::pair<DiscreteGm, std::string> operator()() {
    xt::random::seed(seed);

    auto space = DiscreteSpace(n_variables, n_labels);
    DiscreteGm gm(space);

    // unary factors
    for (std::size_t vi = 0; vi < n_variables; vi++) {
      auto tensor = xt::random::rand<energy_type>({n_labels}, energy_type(-1),
                                                  energy_type(1));
      auto f = std::make_unique<nxtgm::Unary>(tensor);
      auto fid = gm.add_energy_function(std::move(f));
      gm.add_factor({vi}, fid);
    }

    // pairwise potts factors
    for (std::size_t i = 0; i < n_variables - 1; i++) {
      auto beta = xt::random::rand<energy_type>({1}, energy_type(-1),
                                                energy_type(1))(0);
      auto f = std::make_unique<Potts>(n_labels, beta);
    }

    // unsatisfiable constraints
    xt::xarray<energy_type> not_same =
        xt::zeros<energy_type>({n_labels, n_labels});
    xt::xarray<energy_type> not_different =
        xt::ones<energy_type>({n_labels, n_labels});
    for (std::size_t i = 0; i < n_labels; ++i) {
      not_same(i, i) = 1;
      not_different(i, i) = 0;
    }
    auto f_not_same =
        std::make_unique<ArrayDiscreteConstraintFunction>(not_same);
    auto f_not_different =
        std::make_unique<ArrayDiscreteConstraintFunction>(not_different);
    auto fid_not_same = gm.add_constraint_function(std::move(f_not_same));
    auto fid_not_different =
        gm.add_constraint_function(std::move(f_not_different));

    // pairwise constraints along the chain
    for (std::size_t i = 0; i < n_variables - 1; i++) {
      gm.add_constraint({i, i + 1}, fid_not_same);
      gm.add_constraint({i, i + 1}, fid_not_different);
    }

    const std::string name =
        fmt::format("InfesibleModel(n_variables={}, n_labels={}, seed={})",
                    n_variables, n_labels, seed);
    ++seed;
    return std::pair<DiscreteGm, std::string>(std::move(gm), name);
  }

  std::size_t n_variables;
  discrete_label_type n_labels;
  std::uint32_t seed = 0;
};

} // namespace nxtgm::tests
