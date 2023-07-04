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
option(BUILD_TESTS          "Build tests" ON)
option(BUILD_PYTHON         "Build python bindings" ON)
option(BUILD_WITH_COVERAGE  "Build with coverage" OFF)


###############################################################
# dependencies
###############################################################
find_package(Threads REQUIRED)
find_package(fmt REQUIRED)
find_package(Boost REQUIRED)
find_package(xtensor REQUIRED)
find_package(HIGHS REQUIRED)
find_package(nlohmann_json REQUIRED)

###############################################################
# converage
###############################################################
if(BUILD_WITH_COVERAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
endif()


###############################################################
# library
###############################################################

add_library(nxtgm SHARED
    src/nxtgm/energy_functions/discrete_energy_function_base.cpp
    src/nxtgm/energy_functions/discrete_energy_functions.cpp
    src/nxtgm/spaces/discrete_space.cpp
    src/nxtgm/constraint_functions/discrete_constraint_function_base.cpp
    src/nxtgm/constraint_functions/discrete_constraints.cpp
    src/nxtgm/models/solution_value.cpp
    src/nxtgm/models/gm/discrete_gm.cpp
    src/nxtgm/optimizers/gm/discrete/movemaker.cpp
    src/nxtgm/optimizers/gm/discrete/brute_force_naive.cpp
    src/nxtgm/optimizers/gm/discrete/ilp_highs.cpp
    src/nxtgm/optimizers/gm/discrete/dynamic_programming.cpp
    src/nxtgm/optimizers/gm/discrete/icm.cpp
)

###############################################################
# compiler settings / features
###############################################################
target_compile_features(nxtgm PUBLIC cxx_std_17)

###############################################################
# link libraries
###############################################################

target_link_libraries(nxtgm PUBLIC
    fmt::fmt
    Boost::boost
    highs::highs
    xtensor
    nlohmann_json::nlohmann_json
)


###############################################################
# include directories
###############################################################
target_include_directories(nxtgm PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
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
