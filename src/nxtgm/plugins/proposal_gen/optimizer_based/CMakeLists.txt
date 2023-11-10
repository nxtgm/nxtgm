set(PLUGIN_TYPE         proposal_gen)
set(PLUGIN_NAME_CORE    optimizer_based)
set(PLUGIN_TARGET_NAME  ${PLUGIN_TYPE}_${PLUGIN_NAME_CORE})

if(${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    cmake_minimum_required(VERSION 3.12)
    project(nxtgm_${PLUGIN_TYPE}_${PLUGIN_NAME_CORE})
    find_package(nxtgm REQUIRED)
endif()


nxtgm_add_plugin(${PLUGIN_TARGET_NAME} ${PLUGIN_NAME_CORE}.cpp)

# plugin base class / interface headers
target_link_libraries(${PLUGIN_TARGET_NAME}  ${NXTGM_SHARED_LIBRARY})

# install plugin
install(TARGETS ${PLUGIN_TARGET_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/nxtgm/plugins/${PLUGIN_TYPE})

if(NOT ${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    # this is only needed when the plugin is build as part of the whole project
    nxtgm_set_plugin_properties(${PLUGIN_TYPE} ${PLUGIN_TARGET_NAME})
endif()
