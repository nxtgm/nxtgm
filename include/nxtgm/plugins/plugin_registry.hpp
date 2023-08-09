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
    static std::filesystem::path get_plugin_dir();
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
inline std::filesystem::path plugin_registry<FACTORY_BASE>::get_plugin_dir()
{
    std::filesystem::path plugin_dir;
    if (char *plugin_dir_env_value = std::getenv(factory_base_type::plugin_dir_env_var().c_str());
        plugin_dir_env_value != nullptr)
    {
        plugin_dir = std::filesystem::path(plugin_dir_env_value);
    }
    else if (char *plugin_base_dir_env_value = std::getenv("NXTGM_PLUGIN_PATH"); plugin_base_dir_env_value != nullptr)
    {
        plugin_dir = std::filesystem::path(plugin_base_dir_env_value) / factory_base_type::plugin_type();
    }
    else if (char *conda_prefix_env_value = std::getenv("CONDA_PREFIX"); conda_prefix_env_value != nullptr)
    {
        plugin_dir = std::filesystem::path(conda_prefix_env_value) / "lib" / "nxtgm" / "plugins" /
                     factory_base_type::plugin_type();
    }
    else
    {
        throw std::runtime_error(std::string("neither Environment variable ") +
                                 factory_base_type::plugin_dir_env_var() +
                                 std::string(" or NXTGM_PLUGIN_PATH or CONDA_PREFIX is set"));
    }

    if (std::filesystem::exists(plugin_dir) && std::filesystem::is_directory(plugin_dir))
    {
        return plugin_dir;
    }
    else
    {
        const std::string msg = std::string("Path \"") + plugin_dir.string() + std::string("\" for plugin type \"") +
                                factory_base_type::plugin_type() + std::string("\" is not a directory");
        throw std::runtime_error(msg);
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
