enable_testing()


# Prepare doctest for other targets to use
find_package(doctest REQUIRED)



# PLUGIN_PATH_ENV_VARS as one string seperated by ;
set(PLUGIN_ENV_VARS "")

# add env vars for all plugins
SET(PLUGIN_ENV_VARS "${PLUGIN_ENV_VARS};NXTGM_QPBO_PLUGIN_PATH=${QPBO_PLUGIN_OUTPUT_DIRECTORY}")
SET(PLUGIN_ENV_VARS "${PLUGIN_ENV_VARS};NXTGM_MIN_ST_CUT_PLUGIN_PATH=${MIN_ST_CUT_PLUGIN_OUTPUT_DIRECTORY}")


function(add_nxtgm_test target_name)
    message(STATUS "Adding test ${target_name}")
    add_executable(${target_name} main.cpp ${ARGN})
    target_compile_features(${target_name} PRIVATE cxx_std_17)
    target_link_libraries(${target_name} PRIVATE nxtgm doctest::doctest )
    target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME ${target_name} COMMAND ${target_name})
    set_property(TEST ${target_name} PROPERTY ENVIRONMENT "${PLUGIN_ENV_VARS}")
endfunction()

add_nxtgm_test(test_sparse_array                  utils/test_sparse_array.cpp)
add_nxtgm_test(test_solution_value                models/test_solution_value.cpp)
add_nxtgm_test(test_discrete_constraints          constraint_functions/test_discrete_constraints.cpp)
add_nxtgm_test(test_discrete_energy_functions     energy_functions/test_discrete_energy_functions.cpp)
add_nxtgm_test(test_space                         spaces/test_space.cpp)
add_nxtgm_test(test_discrete_gm                   models/gm/test_discrete_gm.cpp)
add_nxtgm_test(test_brute_force_naive             optimizers/gm/discrete/test_brute_force_naive.cpp)
add_nxtgm_test(test_ilp_highs                     optimizers/gm/discrete/test_ilp_highs.cpp)
add_nxtgm_test(test_dynamic_programming           optimizers/gm/discrete/test_dynamic_programming.cpp)
add_nxtgm_test(test_belief_propagation            optimizers/gm/discrete/test_belief_propagation.cpp)
add_nxtgm_test(test_icm                           optimizers/gm/discrete/test_icm.cpp)
add_nxtgm_test(test_matching_icm                  optimizers/gm/discrete/test_matching_icm.cpp)
add_nxtgm_test(test_qpbo                          optimizers/gm/discrete/test_qpbo.cpp)
