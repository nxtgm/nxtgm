# this is needed when the plugin is build on its own
if(${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    cmake_minimum_required(VERSION 3.12)
    project(nxtgm_plugin_qpbo_kolmogorov)
    find_package(nxtgm REQUIRED)
endif()

set(PLUGIN_TYPE         min_st_cut)
set(PLUGIN_TARGET_NAME  ${PLUGIN_TYPE}_kolmogorov)

add_xplugin(${PLUGIN_TARGET_NAME} ${PLUGIN_TARGET_NAME}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/maxflow-v3.04/graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/maxflow-v3.04/maxflow.cpp
)

# plugin base class / interface headers
target_link_libraries(${PLUGIN_TARGET_NAME}  ${NXTGM_SHARED_LIBRARY})

# includes for plugin itself
target_include_directories(${PLUGIN_TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/maxflow-v3.04)

# install plugin
install(TARGETS ${PLUGIN_TARGET_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/nxtgm/plugins/${PLUGIN_TYPE})

if(NOT ${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    # this is only needed when the plugin is build as part of the whole project
    nxtgm_set_plugin_properties(${PLUGIN_TYPE} ${PLUGIN_TARGET_NAME})
endif()
