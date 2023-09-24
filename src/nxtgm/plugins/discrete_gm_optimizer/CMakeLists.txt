set(PLUGIN_TYPE         discrete_gm_optimizer)

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




add_subdirectory(dynamic_programming)
add_subdirectory(graph_cut)
add_subdirectory(icm)
add_subdirectory(matching_icm)
add_subdirectory(qpbo)
add_subdirectory(hqpbo)
add_subdirectory(chained_optimizers)
add_subdirectory(reduced_gm_optimizer)
add_subdirectory(ilp_highs)
add_subdirectory(brute_force_naive)
add_subdirectory(belief_propagation)
