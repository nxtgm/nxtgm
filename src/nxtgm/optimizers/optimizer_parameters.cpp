#include <nxtgm/optimizers/optimizer_parameters.hpp>
#include <sstream>

namespace nxtgm
{

UnknownParameterException::UnknownParameterException(const std::string &str)
    : std::runtime_error("Unknown parameter: " + str)
{
}

OptimizerParameters::Proxy::Proxy(const std::string &key, OptimizerParameters *parent)
    : key(key),
      parent(parent)
{
}

void OptimizerParameters::Proxy::push_back(const std::string &value)
{
    parent->string_parameters[key].push_back(value);
}

void OptimizerParameters::Proxy::push_back(const OptimizerParameters &value)
{
    parent->optimizer_parameters[key].push_back(value);
}

typename OptimizerParameters::Proxy &OptimizerParameters::Proxy::operator=(const std::string &value)
{
    parent->string_parameters[key] = std::vector<std::string>(1, value);
    return *this;
}

typename OptimizerParameters::Proxy &OptimizerParameters::Proxy::operator=(const OptimizerParameters &value)
{
    parent->optimizer_parameters[key] = std::vector<OptimizerParameters>(1, value);
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

void OptimizerParameters::serialize(Serializer &serializer) const
{
    if (any_parameters.size() > 0)
    {
        throw std::runtime_error("any_parameters not supported in serialization");
    }

    serializer(string_parameters);
    serializer(int_parameters);
    serializer(double_parameters);
    serializer(uint64_t(optimizer_parameters.size()));
    for (const auto &[key, optimizer_paramter_vector] : optimizer_parameters)
    {
        serializer(key);
        // vector size
        serializer(uint64_t(optimizer_paramter_vector.size()));
        for (const auto &optimizer_parameter : optimizer_paramter_vector)
        {
            optimizer_parameter.serialize(serializer);
        }
    }
}

OptimizerParameters OptimizerParameters::deserialize(Deserializer &deserializer)
{

    OptimizerParameters p;
    deserializer(p.string_parameters);
    deserializer(p.int_parameters);
    deserializer(p.double_parameters);

    uint64_t optimizer_parameters_size;
    deserializer(optimizer_parameters_size);
    for (uint64_t i = 0; i < optimizer_parameters_size; ++i)
    {
        std::string key;
        deserializer(key);
        uint64_t optimizer_parameter_vector_size;
        deserializer(optimizer_parameter_vector_size);
        for (uint64_t j = 0; j < optimizer_parameter_vector_size; ++j)
        {
            p.optimizer_parameters[key].push_back(OptimizerParameters::deserialize(deserializer));
        }
    }

    return p;
}

// TODO use fking iostream again
template <class STREAM>
STREAM &to_stream(STREAM &out, const OptimizerParameters &p)
{

    // helper to print a vector
    auto print_vector = [&out](const auto &vec) {
        out << "[";
        bool first = true;
        for (const auto &v : vec)
        {
            if (!first)
                out << ", ";
            out << v;
            first = false;
        }
        out << "]";
    };

    out << "OptimizerParameters(";
    bool first = true;
    for (const auto &[key, value] : p.string_parameters)
    {
        if (!first)
            out << ", ";

        out << key << "=";
        print_vector(value);
        first = false;
    }
    for (const auto &[key, value] : p.int_parameters)
    {
        if (!first)
            out << ", ";
        out << key << "=";
        print_vector(value);
        first = false;
    }
    for (const auto &[key, value] : p.double_parameters)
    {
        if (!first)
            out << ", ";
        out << key << "=";
        print_vector(value);
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
        out << key << "=";
        for (const auto &optimizer_parameter : value)
        {
            to_stream(out, optimizer_parameter);
        }
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
        ss << "The following parameters are not supported by '" << optimizer_name << "':\n";
        to_stream(ss, parameters);
        throw UnknownParameterException(ss.str());
    }
}

} // namespace nxtgm
