#pragma once
#include <cstddef>
#include <memory>
#include <string>

#include <nxtgm/nxtgm.hpp>
#include <nxtgm/plugins/plugin.hpp>
#include <nxtgm/plugins/qpbo/qpbo_base.hpp>

#include <nxtgm/models/gm/discrete_gm/gm.hpp>

namespace nxtgm
{

// higher order clique reduction
class HocrBase
{
  public:
    HocrBase() = default;
    virtual ~HocrBase() = default;

    virtual std::size_t add_variable() = 0;
    virtual std::size_t add_variables(std::size_t n) = 0;
    virtual std::size_t num_vars() const = 0;

    virtual void add_term(double coeff, span<const std::size_t> vars) = 0;
    virtual void add_unary_term(double coeff, std::size_t var) = 0;
    virtual void clear() = 0;
    virtual void to_quadratic(QuadraticRepresentationBase *quadratic_representation) = 0;
};

class HocrFactoryBase
{
  public:
    virtual ~HocrFactoryBase() = default;

    static std::string plugin_type()
    {
        return "hocr";
    }

    static std::string plugin_dir_env_var()
    {
        return "NXTGM_HOCR_PLUGIN_PATH";
    }

    // create an instance of the plugin
    virtual std::unique_ptr<HocrBase> create() = 0;

    // priority of the plugin (higher means more important)
    virtual int priority() const = 0;

    // license of the plugin
    virtual std::string license() const = 0;

    // description of the plugin
    virtual std::string description() const = 0;

    // with default implementation
    virtual std::unique_ptr<HocrBase> create(const DiscreteGm &gm);
};

} // namespace nxtgm
