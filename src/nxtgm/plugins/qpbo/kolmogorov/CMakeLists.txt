# this is needed when the plugin is build on its own
if(${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    cmake_minimum_required(VERSION 3.12)
    project(nxtgm_plugin_qpbo_kolmogorov)
    find_package(nxtgm REQUIRED)
    find_package(xplugin REQUIRED)
endif()

set(PLUGIN_TYPE         qpbo)
set(PLUGIN_TARGET_NAME  ${PLUGIN_TYPE}_kolmogorov)

add_xplugin(${PLUGIN_TARGET_NAME} qpbo_kolmogorov.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/QPBO/QPBO.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/QPBO/QPBO_maxflow.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/QPBO/QPBO_postprocessing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/QPBO/QPBO_extra.cpp
)

# plugin base class / interface headers
target_link_libraries(${PLUGIN_TARGET_NAME}  ${NXTGM_SHARED_LIBRARY})

# includes for plugin itself
target_include_directories(${PLUGIN_TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/QPBO)

# install plugin
install(TARGETS ${PLUGIN_TARGET_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/nxtgm/plugins/${PLUGIN_TYPE})

# this is needed when the plugin is build as part of the whole project
if(NOT ${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    if(WIN32)
        set_target_properties(${PLUGIN_TARGET_NAME} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${NXTGM_PLUGIN_OUTPUT_DIRECTORY}/${PLUGIN_TYPE}
            RUNTIME_OUTPUT_DIRECTORY ${NXTGM_PLUGIN_OUTPUT_DIRECTORY}/${PLUGIN_TYPE}
        )
    else()
        set_target_properties(${PLUGIN_TARGET_NAME} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${NXTGM_PLUGIN_OUTPUT_DIRECTORY}/${PLUGIN_TYPE}
        )
    endif()

    set_target_properties(${PLUGIN_TARGET_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${NXTGM_PLUGIN_OUTPUT_DIRECTORY}/${PLUGIN_TYPE})
    add_dependencies(all_plugins ${PLUGIN_TARGET_NAME})
endif()
