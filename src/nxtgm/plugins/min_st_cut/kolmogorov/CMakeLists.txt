set(PLUGIN_TARGET_NAME  min_st_cut_kolmogorov)

find_package(Boost REQUIRED)

add_xplugin(${PLUGIN_TARGET_NAME} ${PLUGIN_TARGET_NAME}.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/maxflow-v3.04/graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/maxflow-v3.04/maxflow.cpp
)

# plugin base class / interface headers
target_include_directories(${PLUGIN_TARGET_NAME} PRIVATE ${NXTGM_INCLUDE_DIR})
target_include_directories(${PLUGIN_TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/maxflow-v3.04)
set_target_properties(${PLUGIN_TARGET_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${NXTGM_MIN_ST_CUT_PLUGIN_OUTPUT_DIRECTORY})
add_dependencies(all_plugins ${PLUGIN_TARGET_NAME})
install(TARGETS ${PLUGIN_TARGET_NAME} DESTINATION ${MIN_ST_CUT_INSTALL_PATH})
