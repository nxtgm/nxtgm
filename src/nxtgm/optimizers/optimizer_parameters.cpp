#include <nxtgm/optimizers/optimizer_parameters.hpp>
#include <ostream>
#include <sstream>

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

bool OptimizerParameters::empty() const
{
    return string_parameters.empty() && int_parameters.empty() && double_parameters.empty() && any_parameters.empty() &&
           optimizer_parameters.empty();
}

std::ostream &operator<<(std::ostream &out, const OptimizerParameters &p)
{
    out << "OptimizerParameters(";
    bool first = true;
    for (const auto &[key, value] : p.string_parameters)
    {
        if (!first)
            out << ", ";
        out << key << "=" << value;
        first = false;
    }
    for (const auto &[key, value] : p.int_parameters)
    {
        if (!first)
            out << ", ";
        out << key << "=" << value;
        first = false;
    }
    for (const auto &[key, value] : p.double_parameters)
    {
        if (!first)
            out << ", ";
        out << key << "=" << value;
        first = false;
    }
    for (const auto &[key, value] : p.any_parameters)
    {
        if (!first)
            out << ", ";
        out << key << "="
            << "<some any value>";
        first = false;
    }
    for (const auto &[key, value] : p.optimizer_parameters)
    {
        if (!first)
            out << ", ";
        out << key << "=" << value;
        first = false;
    }
    out << ")";
    return out;
}

void ensure_all_handled(const std::string &optimizer_name, const OptimizerParameters &parameters)
{
    if (!parameters.empty())
    {
        std::stringstream ss;
        ss << "The following parameters are not supported by the optimizer '" << optimizer_name << "':\n" << parameters;
        throw std::runtime_error(ss.str());
    }
}

} // namespace nxtgm
