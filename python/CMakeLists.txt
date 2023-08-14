SET(PY_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
SET(PY_MODULE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/module)

# dependencies for python bindings
find_package(pybind11 REQUIRED)
find_package(xtensor-python REQUIRED)
find_package(NumPy REQUIRED)
find_package(pybind11_json REQUIRED)

# find numpy
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
                OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE)



# add python bindings target
pybind11_add_module(_nxtgm
    ${PY_SRC_DIR}/nxtgm/models/export_solution_value.cpp
    ${PY_SRC_DIR}/nxtgm/export_nxtgm.cpp
    ${PY_SRC_DIR}/nxtgm/energy_functions/export_discrete_energy_functions.cpp
    ${PY_SRC_DIR}/nxtgm/constraint_functions/export_discrete_constraint_functions.cpp
    ${PY_SRC_DIR}/nxtgm/spaces/export_discrete_space.cpp
    ${PY_SRC_DIR}/nxtgm/models/gm/export_discrete_gm.cpp
    ${PY_SRC_DIR}/nxtgm/optimizers/gm/discrete/export_discrete_gm_optimizers.cpp
)

# link dependencies
target_link_libraries(_nxtgm PRIVATE
    ${NXTGM_SHARED_LIBRARY}
    xtensor-python
)

# include directories
target_include_directories(_nxtgm PRIVATE
    ${NUMPY_INCLUDE_DIR}
)

# copy the pybind11 module to the module directory
add_custom_command(TARGET _nxtgm POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy
                   $<TARGET_FILE:_nxtgm>
                   ${PY_MODULE_DIR}/nxtgm)

# run pytest from cmake
add_custom_target(python_tests
                  ${CMAKE_COMMAND}
                     -E env PYTHONPATH=${PY_MODULE_DIR}
                     ${PYTHON_EXECUTABLE} -m
                     pytest ${CMAKE_CURRENT_SOURCE_DIR}/tests -vvv -s
                  DEPENDS _nxtgm
                  WORKING_DIRECTORY ${PY_MODULE_DIR}
                  COMMENT "Running pytest"
                  USES_TERMINAL
                  )

add_test(NAME python_tests COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target python_tests)


# python get side-packakge dir
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
                OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
                OUTPUT_STRIP_TRAILING_WHITESPACE)


# install the pure python files from  $CMAKE_SOURCE_DIR/python/module/nxtgm to the site-packages directory
install(DIRECTORY ${CMAKE_SOURCE_DIR}/python/module/nxtgm
        DESTINATION ${PYTHON_SITE_PACKAGES}
        FILES_MATCHING PATTERN "*.py"
        PATTERN "__pycache__" EXCLUDE
        )

# install the compiled python module to the site-packages directory
install(TARGETS _nxtgm
        DESTINATION ${PYTHON_SITE_PACKAGES}/nxtgm
        )
