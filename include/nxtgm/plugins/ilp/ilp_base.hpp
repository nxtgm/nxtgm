#pragma once
#include <cstddef>
#include <memory>
#include <string>

#include <nxtgm/nxtgm.hpp>
#include <nxtgm/optimizers/optimizer_base.hpp>
#include <nxtgm/optimizers/optimizer_parameters.hpp>
#include <nxtgm/plugins/ilp/ilp_data.hpp>
#include <nxtgm/plugins/plugin_priority.hpp>
#include <nxtgm/plugins/plugin_registry.hpp>

namespace nxtgm
{

class IlpBase
{
  public:
    IlpBase() = default;
    virtual ~IlpBase() = default;

    virtual OptimizationStatus optimize(const double *starting_point) = 0;
    virtual std::size_t num_variables() const = 0;
    virtual double get_objective_value() = 0;
    virtual void get_solution(double *solution) = 0;
};

class IlpFactoryBase
{

  public:
    class parameters_type
    {
      public:
        inline parameters_type(OptimizerParameters &parameters)
        {
            parameters.assign_and_pop("integer", integer);
            parameters.assign_and_pop("log_level", log_level);
            if (auto it = parameters.int_parameters.find("time_limit_ms"); it != parameters.int_parameters.end())
            {
                time_limit = std::chrono::milliseconds(it->second);
                parameters.int_parameters.erase(it);
            }
        }
        bool integer = true;
        int log_level = 0;
        std::chrono::duration<double> time_limit = std::chrono::duration<double>::max();
    };

    virtual ~IlpFactoryBase() = default;

    inline static std::string plugin_type()
    {
        return "ilp";
    }

    inline static std::string plugin_dir_env_var()
    {
        return "NXTGM_ILP_PLUGIN_PATH";
    }

    // create an instance of the plugin
    virtual std::unique_ptr<IlpBase> create(IlpData &&, OptimizerParameters &&parameters) = 0;

    // priority of the plugin (higher means more important)
    virtual int priority() const = 0;

    // license of the plugin
    virtual std::string license() const = 0;

    // description of the plugin
    virtual std::string description() const = 0;

    static plugin_registry<IlpFactoryBase> &get_registry();
};

} // namespace nxtgm
