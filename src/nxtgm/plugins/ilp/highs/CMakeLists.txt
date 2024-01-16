set(PLUGIN_TYPE         ilp)
set(PLUGIN_NAME_CORE    highs)
set(PLUGIN_TARGET_NAME  ${PLUGIN_TYPE}_${PLUGIN_NAME_CORE})

if(${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    cmake_minimum_required(VERSION 3.12)
    project(nxtgm_${PLUGIN_TYPE}_${PLUGIN_NAME_CORE})
    find_package(nxtgm REQUIRED)
    # append the cmake folder to the module path
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(HIGHS REQUIRED)
message(STATUS "HIGHS_INCLUDE_DIRS: ${HIGHS_INCLUDE_DIRS}")
message(STATUS "HIGHS_LIBRARIES: ${HIGHS_LIBRARIES}")
nxtgm_add_plugin(${PLUGIN_TARGET_NAME} ${PLUGIN_NAME_CORE}.cpp)

# plugin base class / interface headers
target_link_libraries(${PLUGIN_TARGET_NAME}  ${NXTGM_SHARED_LIBRARY} ${HIGHS_LIBRARIES})
target_include_directories(${PLUGIN_TARGET_NAME} PUBLIC ${HIGHS_INCLUDE_DIRS})

# install plugin
install(TARGETS ${PLUGIN_TARGET_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/nxtgm/plugins/${PLUGIN_TYPE})

if(NOT ${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    # this is only needed when the plugin is build as part of the whole project
    nxtgm_set_plugin_properties(${PLUGIN_TYPE} ${PLUGIN_TARGET_NAME})
endif()
