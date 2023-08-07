set(PLUGIN_TARGET_NAME  min_st_cut_boost_boykov_kolmogorov)

find_package(Boost REQUIRED)

add_xplugin(${PLUGIN_TARGET_NAME} boost_boykov_kolmogorov.cpp)

# plugin base class / interface headers
target_include_directories(${PLUGIN_TARGET_NAME} PRIVATE ${NXTGM_INCLUDE_DIR})
# link boost
target_link_libraries(${PLUGIN_TARGET_NAME} boost::boost)

# set output directory for plugin (one level above the current st. all plugins are in the same directory)
set_target_properties(${PLUGIN_TARGET_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${MAX_FLOW_PLUGIN_OUTPUT_DIRECTORY})
# make sure `${PLUGIN_TARGET_NAME}` is built before `all_plugins`
add_dependencies(all_plugins ${PLUGIN_TARGET_NAME})
# install plugin
install(TARGETS ${PLUGIN_TARGET_NAME} DESTINATION ${QPBO_INSTALL_PATH})
