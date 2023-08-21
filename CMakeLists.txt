###############################################################
# general cmake settings
###############################################################
cmake_minimum_required(VERSION 3.12)
project(nxtgm)

# include cmake folder
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

###############################################################
# options
###############################################################
# test related
option(BUILD_TESTS                      "Build tests" ON)
option(BUILD_SINGLE_BINARY_TESTS        "Build tests as single binary" OFF)
# python related
option(BUILD_PYTHON                     "Build python bindings" ON)
# coverage related
option(BUILD_WITH_COVERAGE              "Build with coverage" OFF)
# documentation related
option(BUILD_DOCS                       "Build doxygen documentation" ON)
# plugin related
option(BUILD_PLUGIN_QPBO_KOLMOGOROV     "Build qpbo kolmogorov plugin" ON)

###############################################################
# dependencies
###############################################################
find_package(Threads REQUIRED)
find_package(fmt REQUIRED)
find_package(Boost REQUIRED)
find_package(xtensor REQUIRED)
find_package(HIGHS REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(xplugin REQUIRED)
find_package(tsl-ordered-map REQUIRED)

###############################################################
# converage
###############################################################
if(BUILD_WITH_COVERAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
endif()


###############################################################
# library
###############################################################

add_library(nxtgm_shared SHARED
    src/nxtgm/energy_functions/discrete_energy_function_base.cpp
    src/nxtgm/energy_functions/discrete_energy_functions.cpp
    src/nxtgm/spaces/discrete_space.cpp
    src/nxtgm/constraint_functions/discrete_constraint_function_base.cpp
    src/nxtgm/constraint_functions/discrete_constraints.cpp
    src/nxtgm/models/solution_value.cpp
    src/nxtgm/models/gm/discrete_gm/discrete_gm.cpp
    src/nxtgm/optimizers/optimizer_parameters.cpp
    src/nxtgm/optimizers/gm/discrete/discrete_gm_optimizer_factory.cpp
    src/nxtgm/optimizers/gm/discrete/movemaker.cpp
    src/nxtgm/optimizers/gm/discrete/optimizer_base.cpp
    # this testing needs to be partof the shared library
    # since plugin authors want to test their plugins
    # for emscripten, we might want to remove this (TODO)
    src/nxtgm/models/gm/discrete_gm/testing/testmodels.cpp
    src/nxtgm/models/gm/discrete_gm/testing/optimizer_tester.cpp
)
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
    fmt::fmt
    highs::highs
    Boost::boost
    xtensor
    nlohmann_json::nlohmann_json
)

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
# python
###############################################################
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()


###############################################################
# python
###############################################################
if(BUILD_PYTHON)
    add_subdirectory(python)
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


# install targets
install(TARGETS nxtgm_shared
    EXPORT nxtgmTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

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
