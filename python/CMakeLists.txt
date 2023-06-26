SET(PY_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
SET(PY_MODULE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/module)

# dependencies for python bindings
find_package(pybind11 REQUIRED)
find_package(xtensor-python REQUIRED)
find_package(NumPy REQUIRED)

# find numpy
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
                OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE)



# add python bindings target
pybind11_add_module(_nxtgm  
    ${PY_SRC_DIR}/nxtgm/export_nxtgm.cpp
    ${PY_SRC_DIR}/nxtgm/energy_functions/export_discrete_energy_functions.cpp
    ${PY_SRC_DIR}/nxtgm/spaces/export_discrete_space.cpp
    ${PY_SRC_DIR}/nxtgm/models/gm/export_discrete_gm.cpp
)

# link dependencies
target_link_libraries(_nxtgm PRIVATE 
    nxtgm
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
add_custom_target(pytest
                  ${CMAKE_COMMAND}
                     -E env PYTHONPATH=${PY_MODULE_DIR}  
                     ${PYTHON_EXECUTABLE} -m 
                     pytest ${CMAKE_CURRENT_SOURCE_DIR}/tests -vvv -s
                  DEPENDS _nxtgm
                  WORKING_DIRECTORY ${PY_MODULE_DIR}
                  COMMENT "Running pytest"
                  USES_TERMINAL
                  )