cmake_minimum_required(VERSION 3.12)
# project(nxtgm_javascript_runtime)
# find_package(nxtgm REQUIRED)

option(USE_WASM_EXCEPTIONS "Use wasm exceptions" OFF)
if(USE_WASM_EXCEPTIONS)
    set(EXCEPTION_FLAG "-fwasm-exceptions")
else()
    set(EXCEPTION_FLAG "-fexceptions")
endif()

add_executable(
    nxtgm_javascript_runtime
    ${CMAKE_CURRENT_SOURCE_DIR}/src/export_nxtgm.cpp
)

# link dependencies
target_link_libraries(nxtgm_javascript_runtime PRIVATE
    nxtgm_shared
)

target_include_directories(nxtgm_javascript_runtime PRIVATE
    ${NXTGM_INCLUDE_DIR}
)


target_compile_options(nxtgm_javascript_runtime PRIVATE -fPIC)
target_compile_options(nxtgm_javascript_runtime
        PUBLIC --std=c++17
        PUBLIC -Wno-deprecated
        PUBLIC "SHELL: ${EXCEPTION_FLAG}"
    )

SET(ENVIRONMENT "web,worker")


target_link_options(nxtgm_javascript_runtime
    PUBLIC --bind
    PUBLIC -Wno-unused-command-line-argument
    PUBLIC "SHELL: --bind"
    PUBLIC "SHELL: -s MODULARIZE=1"
    PUBLIC "SHELL: -s EXPORT_NAME=\"nxtgm_create_module\""
    PUBLIC "SHELL: -s EXPORT_ES6=0"
    PUBLIC "SHELL: -s USE_ES6_IMPORT_META=0"
    PUBLIC "SHELL: -s DEMANGLE_SUPPORT=0"
    PUBLIC "SHELL: -s ASSERTIONS=0"
    PUBLIC "SHELL: -s ALLOW_MEMORY_GROWTH=1"
    PUBLIC "SHELL: -s EXIT_RUNTIME=0"
    PUBLIC "SHELL: -s WASM=1"
    PUBLIC "SHELL: -s USE_PTHREADS=0"
    PUBLIC "SHELL: -s ENVIRONMENT=${ENVIRONMENT}"
    PUBLIC "SHELL: ${EXCEPTION_FLAG}"
    PUBLIC "SHELL: -s MAIN_MODULE=1"
    PUBLIC "SHELL: -s TOTAL_STACK=32mb"
    PUBLIC "SHELL: -s INITIAL_MEMORY=64mb"
    PUBLIC "SHELL: -s FORCE_FILESYSTEM"
    PUBLIC "SHELL: -s LZ4=0"
    PUBLIC "SHELL: -flto"
    PUBLIC "SHELL: -s WASM_BIGINT"
    #PUBLIC "SHELL: -s --preload-file  ${CMAKE_BINARY_DIR}/libnxtgm_shared.so@/libnxtgm_shared.so"
    #PUBLIC "SHELL: -s --preload-file ../src/nxtgm/plugins/@/plugins"
    PUBLIC "SHELL: --post-js ${CMAKE_CURRENT_SOURCE_DIR}/src/post_js.js"
)

# install
install(TARGETS nxtgm_javascript_runtime
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES
        "$<TARGET_FILE_DIR:nxtgm_javascript_runtime>/nxtgm_javascript_runtime.wasm"
        DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)
# install(FILES
#         "$<TARGET_FILE_DIR:nxtgm_javascript_runtime>/nxtgm_javascript_runtime.data"
#         DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
# )

# nxtgm_javascript_runtime depends on all_plugins
add_dependencies(nxtgm_javascript_runtime all_plugins)
