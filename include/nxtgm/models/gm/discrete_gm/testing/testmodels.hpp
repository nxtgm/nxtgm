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

#include <cmath>

namespace nxtgm
{

class DiscreteGmTestmodel
{
  public:
    DiscreteGmTestmodel() = default;
    virtual ~DiscreteGmTestmodel() = default;
    virtual std::pair<DiscreteGm, std::string> operator()(unsigned seed) = 0;
};

class Star : public DiscreteGmTestmodel
{
  public:
    Star(std::size_t n_arms, std::size_t n_labels);
    std::pair<DiscreteGm, std::string> operator()(unsigned seed) override;

  private:
    std::size_t n_arms;
    std::size_t n_labels;
};

std::unique_ptr<DiscreteGmTestmodel> star(std::size_t n_arms = 3, std::size_t n_labels = 2);

struct PottsGrid : public DiscreteGmTestmodel
{
  public:
    PottsGrid(std::size_t, std::size_t, discrete_label_type, bool submodular);
    std::pair<DiscreteGm, std::string> operator()(unsigned seed) override;

  private:
    std::size_t n_x;
    std::size_t n_y;
    discrete_label_type n_labels;
    bool submodular;
};

std::unique_ptr<DiscreteGmTestmodel> potts_grid(std::size_t n_x = 2, std::size_t n_y = 2,
                                                discrete_label_type n_labels = 2, bool submodular = false);

class SparsePottsChain : public DiscreteGmTestmodel
{
  public:
    SparsePottsChain(std::size_t n_variables, discrete_label_type n_labels);
    std::pair<DiscreteGm, std::string> operator()(unsigned seed) override;

  private:
    std::size_t n_variables;
    discrete_label_type n_labels;
};

std::unique_ptr<DiscreteGmTestmodel> sparse_potts_chain(std::size_t n_variables = 3, discrete_label_type n_labels = 2);

class PottsChainWithLabelCosts : public DiscreteGmTestmodel
{
  public:
    PottsChainWithLabelCosts(std::size_t n_variables, discrete_label_type n_labels);
    std::pair<DiscreteGm, std::string> operator()(unsigned seed) override;

    std::size_t n_variables;
    discrete_label_type n_labels;
};

std::unique_ptr<DiscreteGmTestmodel> potts_chain_with_label_costs(std::size_t n_variables = 2,
                                                                  discrete_label_type n_labels = 2);

class UniqueLabelChain : public DiscreteGmTestmodel
{
  public:
    UniqueLabelChain(std::size_t n_variables, discrete_label_type n_labels, bool use_pairwise_constraints);
    std::pair<DiscreteGm, std::string> operator()(unsigned seed) override;

  private:
    std::size_t n_variables;
    discrete_label_type n_labels;
    bool use_pairwise_constraints;
};

std::unique_ptr<DiscreteGmTestmodel> unique_label_chain(std::size_t n_variables = 3, discrete_label_type n_labels = 2,
                                                        bool use_pairwise_constraints = false);

class RandomModel : public DiscreteGmTestmodel
{
  public:
    RandomModel(std::size_t n_variables, std::size_t n_factors, std::size_t max_factor_arity,
                discrete_label_type n_labels_max);
    std::pair<DiscreteGm, std::string> operator()(unsigned seed) override;

  private:
    std::size_t n_variables;
    std::size_t n_factors;
    std::size_t max_factor_arity;
    discrete_label_type n_labels_max;
};

std::unique_ptr<DiscreteGmTestmodel> random_model(std::size_t n_variables = 3, std::size_t n_factors = 4,
                                                  std::size_t max_factor_arity = 3,
                                                  discrete_label_type n_labels_max = 3);

class RandomSparseModel : public DiscreteGmTestmodel
{
  public:
    RandomSparseModel(std::size_t n_variables, std::size_t n_factors, std::size_t min_factor_arity,
                      std::size_t max_factor_arity, discrete_label_type n_labels_max, float density);
    std::pair<DiscreteGm, std::string> operator()(unsigned seed) override;

  private:
    std::size_t n_variables;
    std::size_t n_factors;
    std::size_t min_factor_arity;
    std::size_t max_factor_arity;
    discrete_label_type n_labels_max;
    float density;
};

std::unique_ptr<DiscreteGmTestmodel> random_sparse_model(std::size_t n_variables = 3, std::size_t n_factors = 4,
                                                         std::size_t min_factor_arity = 1,
                                                         std::size_t max_factor_arity = 3,
                                                         discrete_label_type n_labels_max = 3, float density = 0.1f);

class InfeasibleModel : public DiscreteGmTestmodel
{
  public:
    InfeasibleModel(std::size_t n_variables, discrete_label_type n_labels);
    std::pair<DiscreteGm, std::string> operator()(unsigned seed) override;

  private:
    std::size_t n_variables;
    discrete_label_type n_labels;
};

std::unique_ptr<DiscreteGmTestmodel> infeasible_model(std::size_t n_variables = 3, discrete_label_type n_labels = 2);

} // namespace nxtgm