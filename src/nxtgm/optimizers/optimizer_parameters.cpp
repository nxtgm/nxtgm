#include <nxtgm/optimizers/optimizer_parameters.hpp>

namespace nxtgm
{
OptimizerParameters::Proxy::Proxy(const std::string &key, OptimizerParameters *parent)
    : key(key),
      parent(parent)
{
}

typename OptimizerParameters::Proxy &OptimizerParameters::Proxy::operator=(const std::string &value)
{
    parent->string_parameters[key] = value;
    return *this;
}

typename OptimizerParameters::Proxy &OptimizerParameters::Proxy::operator=(const OptimizerParameters &value)
{
    parent->optimizer_parameters[key] = value;
    return *this;
}

typename OptimizerParameters::Proxy OptimizerParameters::operator[](const std::string &key)
{
    return Proxy(key, this);
}

} // namespace nxtgm
