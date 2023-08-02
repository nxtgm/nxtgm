enable_testing()


# Prepare doctest for other targets to use
find_package(doctest REQUIRED)

# Make test executable
add_executable(cpp_tests main.cpp
    # test_version.cpp
    utils/test_sparse_array.cpp
    # models/test_solution_value.cpp
    constraint_functions/test_discrete_constraints.cpp
    energy_functions/test_discrete_energy_functions.cpp
    spaces/test_space.cpp
    # models/gm/test_discrete_gm.cpp
    optimizers/gm/discrete/test_brute_force_naive.cpp
    optimizers/gm/discrete/test_ilp_highs.cpp
    optimizers/gm/discrete/test_dynamic_programming.cpp
    optimizers/gm/discrete/test_belief_propagation.cpp
    optimizers/gm/discrete/test_icm.cpp
    optimizers/gm/discrete/test_matching_icm.cpp
)
target_compile_features(cpp_tests PRIVATE cxx_std_17)
target_link_libraries(cpp_tests PRIVATE nxtgm doctest::doctest )
target_include_directories(cpp_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# add dependencies on `tests` target
add_test(NAME cpp_tests COMMAND cpp_tests)
