set(PLUGIN_TARGET_NAME  qpbo_kolmogorov)

add_xplugin(${PLUGIN_TARGET_NAME} qpbo_kolmogorov.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/QPBO/QPBO.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/QPBO/QPBO_maxflow.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/QPBO/QPBO_postprocessing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/QPBO/QPBO_extra.cpp
)

# plugin base class / interface headers
target_include_directories(${PLUGIN_TARGET_NAME} PRIVATE ${NXTGM_INCLUDE_DIR})
# includes for plugin itself
target_include_directories(${PLUGIN_TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/QPBO)
# set output directory for plugin (one level above the current st. all plugins are in the same directory)
set_target_properties(${PLUGIN_TARGET_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${QPBO_PLUGIN_OUTPUT_DIRECTORY})
# make sure `${PLUGIN_TARGET_NAME}` is built before `all_plugins`
add_dependencies(all_plugins ${PLUGIN_TARGET_NAME})
# install plugin
install(TARGETS ${PLUGIN_TARGET_NAME} DESTINATION ${QPBO_INSTALL_PATH})
