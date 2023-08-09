#pragma once

#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <xplugin/xplugin_registry.hpp>

namespace nxtgm
{

template <class FACTORY_BASE>
class plugin_registry : public xp::xthread_save_plugin_registry<FACTORY_BASE>
{

  public:
    using factory_base_type = FACTORY_BASE;

    plugin_registry();
    factory_base_type *highest_priority_factory() const;

    factory_base_type *get_factory(const std::string &plugin_name);

  private:
    static std::string get_plugin_dir();
    factory_base_type *m_highest_priority_factory = nullptr;
};

template <class FACTORY_BASE>
plugin_registry<FACTORY_BASE>::plugin_registry()
    : xp::xthread_save_plugin_registry<factory_base_type>(get_plugin_dir())
{
    if (!this->empty())
    {
        int highest_priority = (std::numeric_limits<int>::min)();
        for (auto &[plugin_name, factory] : *this)
        {
            if (factory->priority() > highest_priority)
            {
                highest_priority = factory->priority();
                m_highest_priority_factory = factory;
            }
        }
    }
}

template <class FACTORY_BASE>
typename plugin_registry<FACTORY_BASE>::factory_base_type *plugin_registry<FACTORY_BASE>::highest_priority_factory()
    const
{
    return m_highest_priority_factory;
}

template <class FACTORY_BASE>
inline plugin_registry<FACTORY_BASE> &get_plugin_registry()
{
    static plugin_registry<FACTORY_BASE> registry;
    return registry;
}

template <class FACTORY_BASE>
inline std::string plugin_registry<FACTORY_BASE>::get_plugin_dir()
{
    if (const char *path = std::getenv(factory_base_type::plugin_dir_env_var().c_str()); path != nullptr)
    {
        if (std::filesystem::exists(path) && std::filesystem::is_directory(path))
        {
            std::cout << "\nUSING PLUGIN DIR: " << path << std::endl;
            return std::string(path);
        }
        else
        {
            throw std::runtime_error(std::string("Path \"") + std::string(path) + "\" from environment variable " +
                                     factory_base_type::plugin_dir_env_var() + " is not a directory");
        }
    }
    else
    {
        throw std::runtime_error("Environment variable " + factory_base_type::plugin_dir_env_var() + " not set");
    }
}
template <class FACTORY_BASE>
typename plugin_registry<FACTORY_BASE>::factory_base_type *plugin_registry<FACTORY_BASE>::get_factory(
    const std::string &plugin_name)
{
    if (plugin_name.empty())
    {
        if (this->empty())
        {
            throw std::runtime_error("No plugins found for " + factory_base_type::plugin_type());
        }
        return this->highest_priority_factory();
    }
    else if (this->contains(plugin_name))
    {
        return this->operator[](plugin_name);
    }
    else
    {
        throw std::runtime_error(factory_base_type::plugin_type() + " " + plugin_name + " not found");
    }
}

} // namespace nxtgm
