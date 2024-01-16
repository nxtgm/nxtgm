# FindHIGHS.cmake

find_path(HIGHS_INCLUDE_DIR NAMES Highs.h PATHS $ENV{CONDA_PREFIX}/include/highs)

message(STATUS "HIGHS_INCLUDE_DIR: ${HIGHS_INCLUDE_DIR}")

find_library(HIGHS_LIBRARY NAMES highs PATHS $ENV{CONDA_PREFIX}/lib)


message(STATUS "HIGHS_LIBRARY: ${HIGHS_LIBRARY}")

if (HIGHS_INCLUDE_DIR AND HIGHS_LIBRARY)
    set(HIGHS_FOUND TRUE)
endif (HIGHS_INCLUDE_DIR AND HIGHS_LIBRARY)

if (HIGHS_FOUND)
    if (NOT HIGHS_FIND_QUIETLY)
        message(STATUS "Found HIGHS: ${HIGHS_LIBRARY}")
    endif (NOT HIGHS_FIND_QUIETLY)
else (HIGHS_FOUND)
    if (HIGHS_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find HIGHS")
    endif (HIGHS_FIND_REQUIRED)
endif (HIGHS_FOUND)

if (HIGHS_FOUND)
    set(HIGHS_LIBRARIES ${HIGHS_LIBRARY})
    set(HIGHS_INCLUDE_DIRS ${HIGHS_INCLUDE_DIR})
endif (HIGHS_FOUND)
