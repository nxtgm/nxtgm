###############################################################
# general cmake settings
###############################################################
cmake_minimum_required(VERSION 3.12)
project(nxtgm VERSION 0.1.0)

# include cmake folder
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")


# helper function to add plugins
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/nxtgm_add_nxtgm_plugin.cmake)

###############################################################
# options
###############################################################
# test related
option(BUILD_TESTS                      "Build tests" ON)
option(BUILD_SINGLE_BINARY_TESTS        "Build tests as single binary" OFF)
# python-bindings related
option(BUILD_PYTHON_BINDINGS            "Build python bindings" ON)
# python-bindings related
option(BUILD_JAVASCRIPT_BINDINGS        "Build javascritp bindings" OFF)

# coverage related
option(BUILD_WITH_COVERAGE              "Build with coverage" OFF)
# documentation related
option(BUILD_DOCS                       "Build doxygen documentation" ON)

# install related
option(INSTALL_DEV                      "Install dev files" ON)
option(INSTALL_LIBRARY                 "Install library" ON)

# plugin related
option(BUILD_PLUGINS_DISCRETE_GM_OPTIMIZER                   "Build *ALL* discrete gm optimizer plugins" ON)
option(BUILD_PLUGIN_QPBO_KOLMOGOROV                          "Build qpbo kolmogorov plugin" ON)
option(BUILD_PLUGIN_HOCR_FIX                                 "Build fix higher order qpbo plugin" ON)
option(BUILD_PLUGIN_MIN_ST_CUT_KOLMOGOROV                    "Build kolmogorov min st cut plugin" ON)


###############################################################
# dependencies
###############################################################
find_package(Threads REQUIRED)
find_package(xtensor REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(xplugin REQUIRED)
find_package(tsl-ordered-map REQUIRED)
find_package(tl-expected REQUIRED)

###############################################################
# converage
###############################################################
if(BUILD_WITH_COVERAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
endif()

###############################################################
# compile / platform settings
###############################################################
if(EMSCRIPTEN)
    message(STATUS "Building for emscripten")
    set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)
    set(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS   "-s SIDE_MODULE=1 -s WASM_BIGINT")
    set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "-s SIDE_MODULE=1 -s WASM_BIGINT")
    set(CMAKE_STRIP FALSE)  # used by default in pybind11 on .so modules
endif()

if(WIN32)
    # disable min max macros
    add_definitions(-DNOMINMAX)
endif()

###############################################################
# library
###############################################################

SET(NXTGM_LIBRARY_SOURCES
    src/nxtgm/utils/timer.cpp
    # src/nxtgm/utils/serialize.cpp
    src/nxtgm/functions/argmin2.cpp
    src/nxtgm/functions/discrete_function_base.cpp
    src/nxtgm/functions/discrete_energy_function_base.cpp
    src/nxtgm/functions/xarray_energy_function.cpp
    src/nxtgm/functions/sparse_energy_function.cpp
    src/nxtgm/functions/label_costs_energy_function.cpp
    src/nxtgm/functions/potts_energy_function.cpp
    src/nxtgm/functions/discrete_constraint_function_base.cpp
    src/nxtgm/functions/array_constraint_function.cpp
    src/nxtgm/functions/unique_labels_constraint_function.cpp
    src/nxtgm/spaces/discrete_space.cpp
    src/nxtgm/models/solution_value.cpp
    src/nxtgm/models/gm/discrete_gm/discrete_gm.cpp
    src/nxtgm/optimizers/optimizer_parameters.cpp
    src/nxtgm/optimizers/gm/discrete/discrete_gm_optimizer_factory.cpp
    src/nxtgm/optimizers/gm/discrete/movemaker.cpp
    src/nxtgm/optimizers/gm/discrete/optimizer_base.cpp
    src/nxtgm/optimizers/gm/discrete/fusion.cpp
    src/nxtgm/plugins/hocr/hocr_base.cpp
    src/nxtgm/plugins/ilp/ilp_base.cpp
    src/nxtgm/plugins/min_st_cut/min_st_cut_base.cpp
    src/nxtgm/plugins/proposal_gen/proposal_gen_base.cpp
    src/nxtgm/plugins/qpbo/qpbo_base.cpp
    src/nxtgm/plugins/plugin_registry.cpp
)
if(NOT EMSCRIPTEN)
    SET(NXTGM_LIBRARY_SOURCES ${NXTGM_LIBRARY_SOURCES}
        # this testing needs to be partof the shared library
        # since plugin authors want to test their plugins
        src/nxtgm/models/gm/discrete_gm/testing/testmodels.cpp
        src/nxtgm/models/gm/discrete_gm/testing/optimizer_tester.cpp
    )
endif()
add_library(nxtgm_shared SHARED
    ${NXTGM_LIBRARY_SOURCES}
)
if(EMSCRIPTEN)
    # add defintion NXTGM_NO_THREADS
    target_compile_definitions(nxtgm_shared PUBLIC NXTGM_NO_THREADS)
elseif(WIN32)
    target_compile_definitions(nxtgm_shared PUBLIC NOMINMAX)
endif()
SET(NXTGM_SHARED_LIBRARY nxtgm_shared)

###############################################################
# include directories
###############################################################
SET(NXTGM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(nxtgm_shared PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)


###############################################################
# plugins
###############################################################
SET(NXTGM_PLUGIN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/nxtgm/plugins)
add_subdirectory(src/nxtgm/plugins)


###############################################################
# compiler settings / features
###############################################################
target_compile_features(nxtgm_shared PUBLIC cxx_std_17)
if (WIN32)
    set_property(TARGET nxtgm_shared PROPERTY WINDOWS_EXPORT_ALL_SYMBOLS true)
endif()

###############################################################
# link libraries
###############################################################

# public
target_link_libraries(nxtgm_shared PUBLIC
    xtensor
    nlohmann_json::nlohmann_json
    tl::expected
)
if(CMAKE_DL_LIBS)
    target_link_libraries(nxtgm_shared PUBLIC ${CMAKE_DL_LIBS})
endif()

# private
target_link_libraries(nxtgm_shared PRIVATE
    xplugin::xplugin
    tsl::ordered_map
)



###############################################################
#  tests
###############################################################
# cmake use tests
enable_testing()


###############################################################
# tests
###############################################################
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()


###############################################################
# python-bindings
###############################################################
if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()

###############################################################
# javascript-bindings
###############################################################
if(BUILD_JAVASCRIPT_BINDINGS)
    if(NOT EMSCRIPTEN)
        message(FATAL_ERROR "Javascript bindings can only be built with emscripten")
    endif()
    add_subdirectory(javascript)
endif()



###############################################################
# docs
###############################################################
if(BUILD_DOCS)
    add_subdirectory(docs)
endif()


###############################################################
# install
###############################################################


if (APPLE)
    set_target_properties(nxtgm_shared PROPERTIES
        MACOSX_RPATH ON
    )

    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib; ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
elseif(NOT EMSCRIPTEN)
    set_target_properties(nxtgm_shared PROPERTIES
        BUILD_WITH_INSTALL_RPATH 1
    )

    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib; ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
    set(CMAKE_SKIP_BUILD_RPATH FALSE)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()




# gnu install dirs
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(NXTGM_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}" CACHE STRING "install path for nxtgmConfig.cmake")

if(INSTALL_LIBRARY)
    # install targets
    install(TARGETS nxtgm_shared
        EXPORT nxtgmTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
endif()

if(INSTALL_DEV)

    # install the targets file
    INSTALL(
        EXPORT nxtgmTargets
        FILE nxtgmTargets.cmake
        DESTINATION "${NXTGM_CMAKECONFIG_INSTALL_DIR}"
        NAMESPACE nxtgm::
    )

    # configure the config file
    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/nxtgmConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/nxtgmConfig.cmake
        INSTALL_DESTINATION ${NXTGM_CMAKECONFIG_INSTALL_DIR}
    )

    # the config
    install(
        FILES "${CMAKE_CURRENT_BINARY_DIR}/nxtgmConfig.cmake"
        DESTINATION "${NXTGM_CMAKECONFIG_INSTALL_DIR}"
    )

    # install headers
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/nxtgm
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    # install cmake file
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/nxtgm_add_nxtgm_plugin.cmake
        DESTINATION ${NXTGM_CMAKECONFIG_INSTALL_DIR})

endif()
