function(nxtgm_add_plugin target_name)

    add_xplugin(${target_name} ${ARGN})

    if(EMSCRIPTEN)
        message(STATUS "Emscripten detected, adding sWASM_BIGINT to target ${target_name}")
        target_link_options(${target_name}
            PUBLIC "SHELL: -sWASM_BIGINT"
        )
    endif()

endfunction()


function(nxtgm_set_plugin_flags)
    if(EMSCRIPTEN)
        message(STATUS "Building for emscripten")
        set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)
        set(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS   "-s SIDE_MODULE=1 -s WASM_BIGINT")
        set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "-s SIDE_MODULE=1 -s WASM_BIGINT")
        set(CMAKE_STRIP FALSE)  # used by default in pybind11 on .so modules
    endif()

endfunction()
