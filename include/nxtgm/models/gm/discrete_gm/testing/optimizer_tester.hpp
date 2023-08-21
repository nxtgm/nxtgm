#pragma once

#include <nxtgm/models/gm/discrete_gm/testing/testmodels.hpp>
#include <nxtgm/nxtgm.hpp>
#include <string>

#include <nxtgm/optimizers/gm/discrete/discrete_gm_optimizer_factory.hpp>
#include <nxtgm/utils/tuple_for_each.hpp>

namespace nxtgm
{
std::pair<typename DiscreteGm::solution_type, SolutionValue> solve_brute_force(const DiscreteGm &model);

class DiscreteGmOptimizerRequireBase
{
  public:
    virtual ~DiscreteGmOptimizerRequireBase() = default;
    virtual std::string name() const = 0;
    virtual void require(DiscreteGmOptimizerBase *optimizer, OptimizationStatus status,
                         const std::string &info) const = 0;
};

class RequireOptimizationStatus : public DiscreteGmOptimizerRequireBase
{
  public:
    RequireOptimizationStatus(OptimizationStatus should_status);
    virtual ~RequireOptimizationStatus() = default;
    std::string name() const override;
    void require(DiscreteGmOptimizerBase *optimizer, OptimizationStatus status, const std::string &info) const override;

  private:
    OptimizationStatus should_status;
};
std::unique_ptr<DiscreteGmOptimizerRequireBase> require_optimization_status(OptimizationStatus should_status);

class RequireFeasiblity : public DiscreteGmOptimizerRequireBase
{
  public:
    virtual ~RequireFeasiblity() = default;
    std::string name() const override;
    void require(DiscreteGmOptimizerBase *optimizer, OptimizationStatus status, const std::string &info) const override;
};
std::unique_ptr<DiscreteGmOptimizerRequireBase> require_feasiblity();

class RequireOptimality : public DiscreteGmOptimizerRequireBase
{
  public:
    RequireOptimality(bool proven);
    virtual ~RequireOptimality() = default;
    std::string name() const override;
    void require(DiscreteGmOptimizerBase *optimizer, OptimizationStatus status, const std::string &info) const override;

  private:
    bool proven = false;
};
std::unique_ptr<DiscreteGmOptimizerRequireBase> require_optimality(bool proven = false);

class RequireInfesibility : public DiscreteGmOptimizerRequireBase
{
  public:
    RequireInfesibility(bool proven);
    virtual ~RequireInfesibility() = default;
    std::string name() const override;
    void require(DiscreteGmOptimizerBase *optimizer, OptimizationStatus status, const std::string &info) const override;
    bool proven = true;
};
std::unique_ptr<DiscreteGmOptimizerRequireBase> require_infesibility(bool proven = true);

class RequireLocalOptimality : public DiscreteGmOptimizerRequireBase
{
  public:
    virtual ~RequireLocalOptimality() = default;
    RequireLocalOptimality(bool proven = false);
    std::string name() const override;
    void require(DiscreteGmOptimizerBase *optimizer, OptimizationStatus status, const std::string &info) const override;

  private:
    bool proven;
};
std::unique_ptr<DiscreteGmOptimizerRequireBase> require_local_optimality(bool proven = false);

class RequireLocalNOptimality : public DiscreteGmOptimizerRequireBase
{
  public:
    RequireLocalNOptimality(std::size_t n);
    virtual ~RequireLocalNOptimality() = default;
    std::string name() const override;
    void require(DiscreteGmOptimizerBase *optimizer, OptimizationStatus status, const std::string &info) const override;
    std::size_t n;
};
std::unique_ptr<DiscreteGmOptimizerRequireBase> require_local_n_optimality(std::size_t n = 2);

struct TestDiscreteGmOptimizerOptions
{
    static std::chrono::duration<double> default_per_model_time_limit();
    unsigned seed = 0;
    bool verbose = false;
    bool with_testing_callback = true;
    std::chrono::duration<double> per_model_time_limit = default_per_model_time_limit();
};

void test_discrete_gm_optimizer(const std::string optimizer_name, const OptimizerParameters &parameters,
                                std::initializer_list<std::unique_ptr<DiscreteGmTestmodel>> model_generators,
                                std::initializer_list<std::unique_ptr<DiscreteGmOptimizerRequireBase>> requirements,
                                const TestDiscreteGmOptimizerOptions &options = TestDiscreteGmOptimizerOptions());

void test_discrete_gm_optimizer(const std::string optimizer_name, const OptimizerParameters &parameters,
                                std::unique_ptr<DiscreteGmTestmodel> model_generator,
                                std::initializer_list<std::unique_ptr<DiscreteGmOptimizerRequireBase>> requirements,
                                const TestDiscreteGmOptimizerOptions &options = TestDiscreteGmOptimizerOptions());

void test_discrete_gm_optimizer(const std::string optimizer_name, const OptimizerParameters &parameters,
                                std::initializer_list<std::unique_ptr<DiscreteGmTestmodel>> model_generators,
                                std::unique_ptr<DiscreteGmOptimizerRequireBase> requirement,
                                const TestDiscreteGmOptimizerOptions &options = TestDiscreteGmOptimizerOptions());

void test_discrete_gm_optimizer(const std::string optimizer_name, const OptimizerParameters &parameters,
                                std::unique_ptr<DiscreteGmTestmodel> model_generator,
                                std::unique_ptr<DiscreteGmOptimizerRequireBase> requirement,
                                const TestDiscreteGmOptimizerOptions &options = TestDiscreteGmOptimizerOptions());

}; // namespace nxtgm
