# Prepare doctest for other targets to use
find_package(doctest REQUIRED)

# Make test executable
add_executable(tests main.cpp
    # test_version.cpp
    # models/test_solution_value.cpp
    # constraint_functions/test_discrete_constraints.cpp
    energy_functions/test_discrete_energy_functions.cpp
    spaces/test_space.cpp
    # models/gm/test_discrete_gm.cpp
    optimizers/gm/discrete/test_brute_force_naive.cpp
    optimizers/gm/discrete/test_ilp_highs.cpp
)
target_compile_features(tests PRIVATE cxx_std_17)
target_link_libraries(tests PRIVATE nxtgm doctest::doctest )
target_include_directories(tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})