#include <memory>
#include <nxtgm/plugins/hocr/hocr_base.hpp>
#include <nxtgm/plugins/qpbo/qpbo_base.hpp>
#include <string>

// xplugin
#include "higher_order_energy/higher-order-energy.hpp"
#include <xplugin/xfactory.hpp>

#include <iostream>

namespace nxtgm
{

class QuadraticRepresetationWrapper
{

  public:
    using NodeId = std::size_t;

    QuadraticRepresetationWrapper(QuadraticRepresentationBase *quadratic_representation)
        : quadratic_representation_(quadratic_representation)
    {
    }

    int AddNode(int n = 1)
    {
        return quadratic_representation_->add_nodes(std::size_t(n));
    }

    void SetMaxEdgeNum(int n)
    {
        quadratic_representation_->set_max_edge_num(std::size_t(n));
    }
    void AddUnaryTerm(int n, double coeff0, double coeff1)
    {
        double coeffs[2] = {coeff0, coeff1};
        quadratic_representation_->add_unary_term(std::size_t(n), coeffs);
    }
    void AddPairwiseTerm(int n1, int n2, double E00, double E01, double E10, double E11)
    {
        double coeffs[4] = {E00, E01, E10, E11};
        quadratic_representation_->add_pairwise_term(std::size_t(n1), std::size_t(n2), coeffs);
    }

  private:
    QuadraticRepresentationBase *quadratic_representation_;
};

class HocrFix : public HocrBase
{

  public:
    ~HocrFix() = default;
    HocrFix()
        : HocrBase()
    {
    }

    std::size_t add_variable() override
    {
        return static_cast<std::size_t>(higher_order_energy_.AddVar());
    }

    std::size_t add_variables(std::size_t n) override
    {
        return static_cast<std::size_t>(higher_order_energy_.AddVars(int(n)));
    }

    std::size_t num_vars() const override
    {
        return static_cast<std::size_t>(higher_order_energy_.NumVars());
    }

    void add_term(double coeff, span<const std::size_t> vars) override
    {
        int int_vars[10];
        std::copy(vars.begin(), vars.end(), int_vars);
        higher_order_energy_.AddTerm(coeff, vars.size(), int_vars);
    }

    void add_unary_term(double coeff, std::size_t var) override
    {
        higher_order_energy_.AddUnaryTerm(int(var), coeff);
    }

    void clear() override
    {
        higher_order_energy_.Clear();
    }

    void to_quadratic(QuadraticRepresentationBase *quadratic_representation) override
    {
        QuadraticRepresetationWrapper wrapper(quadratic_representation);
        higher_order_energy_.ToQuadratic(wrapper);
    }

  private:
    HigherOrderEnergy<double, 10> higher_order_energy_;
};

class HocrFixFactory : public HocrFactoryBase
{
  public:
    using factory_base_type = HocrFactoryBase;
    HocrFixFactory() = default;
    ~HocrFixFactory() = default;

    std::unique_ptr<HocrBase> create() override
    {
        return std::make_unique<HocrFix>();
    }

    int priority() const override
    {
        return nxtgm::plugin_priority(PluginPriority::HIGH);
    }

    std::string license() const override
    {
        return "MIT";
    }
    std::string description() const override
    {
        return "Higher order clique reduction by Alexander Fix";
    }
};

} // namespace nxtgm

XPLUGIN_CREATE_XPLUGIN_FACTORY(nxtgm::HocrFixFactory);
