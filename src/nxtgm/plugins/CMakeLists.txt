add_custom_target(all_plugins)

function(nxtgm_set_plugin_properties PLUGIN_TYPE PLUGIN_TARGET_NAME)
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
endfunction()


option(BUILD_PLUGINS                        "Build any plugin" ON)
if(BUILD_PLUGINS)
    add_subdirectory(min_st_cut)
    add_subdirectory(qpbo)
    add_subdirectory(hocr)
    add_subdirectory(discrete_gm_optimizer)
    add_subdirectory(proposal_gen)
    add_subdirectory(ilp)
endif()
