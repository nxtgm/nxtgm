enable_testing()


# Prepare doctest for other targets to use
find_package(doctest REQUIRED)


if(BUILD_SINGLE_BINARY_TESTS)
    # in case of a single binary we collect all sources
    SET(ALL_TEST_SOURCES "")
endif()

message(STATUS "PLUGIN_ENV_VARS: ${PLUGIN_ENV_VARS}")

# function to add a test binary
function(add_nxtgm_test name)
    set(target_name nxtgm_test_${name})
    message(STATUS "Adding test ${target_name}")
    add_executable(${target_name} main.cpp ${ARGN})
    target_compile_features(${target_name} PRIVATE cxx_std_17)
    target_link_libraries(${target_name} PRIVATE nxtgm_shared doctest::doctest )
    target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME ${target_name} COMMAND ${target_name})
    set_property(TEST ${target_name} PROPERTY ENVIRONMENT "NXTGM_PLUGIN_PATH=${NXTGM_PLUGIN_OUTPUT_DIRECTORY}")
endfunction()

# function to register a test
# this will either add the test to the ALL_TEST_SOURCES list
# or add the test directly depending on BUILD_SINGLE_BINARY_TESTS
function(nxtgm_regiter_test target_name)
    if(BUILD_SINGLE_BINARY_TESTS)
        set(ALL_TEST_SOURCES ${ALL_TEST_SOURCES} ${ARGN} PARENT_SCOPE)
    else()
        add_nxtgm_test(${target_name} ${ARGN})
    endif()
endfunction()


nxtgm_regiter_test(sparse_array                  utils/test_sparse_array.cpp)
nxtgm_regiter_test(solution_value                models/test_solution_value.cpp)
nxtgm_regiter_test(discrete_constraints          constraint_functions/test_discrete_constraints.cpp)
nxtgm_regiter_test(discrete_energy_functions     energy_functions/test_discrete_energy_functions.cpp)
nxtgm_regiter_test(space                         spaces/test_space.cpp)
nxtgm_regiter_test(discrete_gm                   models/gm/test_discrete_gm.cpp)
nxtgm_regiter_test(brute_force_naive             optimizers/gm/discrete/test_brute_force_naive.cpp)
nxtgm_regiter_test(ilp_highs                     optimizers/gm/discrete/test_ilp_highs.cpp)
nxtgm_regiter_test(dynamic_programming           optimizers/gm/discrete/test_dynamic_programming.cpp)
nxtgm_regiter_test(belief_propagation            optimizers/gm/discrete/test_belief_propagation.cpp)
nxtgm_regiter_test(icm                           optimizers/gm/discrete/test_icm.cpp)
nxtgm_regiter_test(matching_icm                  optimizers/gm/discrete/test_matching_icm.cpp)
nxtgm_regiter_test(qpbo                          optimizers/gm/discrete/test_qpbo.cpp)


if(BUILD_SINGLE_BINARY_TESTS)
    message(STATUS "Adding single binary test")
    message(STATUS "ALL_TEST_SOURCES: ${ALL_TEST_SOURCES}")
    add_nxtgm_test(all ${ALL_TEST_SOURCES})

    # install test binary
    install(TARGETS nxtgm_test_all DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
