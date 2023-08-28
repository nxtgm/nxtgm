set(PLUGIN_TYPE         discrete_gm_optimizer)
SET(DISCRETE_GM_OPTIMIZER_INSTALL_PATH lib/nxtgm/plugins/${PLUGIN_TYPE})


function(add_solver BASE_NAME)
    SET(PLUGIN_TARGET_NAME discrete_gm_optimizer_${BASE_NAME})
    add_xplugin(${PLUGIN_TARGET_NAME} ${BASE_NAME}.cpp)
    target_link_libraries(${PLUGIN_TARGET_NAME}  ${NXTGM_SHARED_LIBRARY})
    install(TARGETS ${PLUGIN_TARGET_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/nxtgm/plugins/${PLUGIN_TYPE})


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
    add_dependencies(all_plugins ${PLUGIN_TARGET_NAME})

    # this is only needed when the plugin is build as part of the whole project
    nxtgm_set_plugin_properties(${PLUGIN_TYPE} ${PLUGIN_TARGET_NAME})

endfunction()


add_solver(belief_propagation)
add_solver(brute_force_naive)
add_solver(dynamic_programming)
add_solver(graph_cut)
add_solver(icm)
add_solver(ilp_highs)
add_solver(matching_icm)
add_solver(qpbo)
add_solver(chained_optimizers)
add_solver(reduced_gm_optimizer)
