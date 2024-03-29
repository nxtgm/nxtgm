# FindHIGHS.cmake

if(NOT HIGHS_INCLUDE_DIR)

    find_path(HIGHS_INCLUDE_DIR NAMES Highs.h
    HINTS
        $ENV{CONDA_PREFIX}/include/highs
        $ENV{PREFIX}/include
    PATH
        $ENV{CONDA_PREFIX}/include/highs
        $ENV{PREFIX}/include
    )
endif()

if(NOT HIGHS_LIBRARY)
    find_library(HIGHS_LIBRARY NAMES highs
    HINTS
        $ENV{CONDA_PREFIX}/lib
        $ENV{PREFIX}/lib
    PATH
        $ENV{CONDA_PREFIX}/lib
        $ENV{PREFIX}/lib
    )
endif()


# handle the QUIETLY and REQUIRED arguments and set HIGHS_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(HIGHS DEFAULT_MSG
  HIGHS_INCLUDE_DIR
  HIGHS_LIBRARY
)

IF(HIGHS_FOUND)
  SET(HIGHS_LIBRARIES ${HIGHS_LIBRARY} )
  SET(HIGHS_INCLUDE_DIRS ${HIGHS_INCLUDE_DIR} )
ELSE()
  SET(HIGHS_INCLUDE_DIR                   HIGHS_INCLUDE_DIR-NOTFOUND)
  SET(HIGHS_LIBRARY             		  HIGHS_LIBRARY-NOTFOUND)
  SET(HIGHS_LIBRARIES                     HIGHS_LIBRARIES-NOTFOUND)
  SET(HIGHS_INCLUDE_DIRS                  HIGHS_INCLUDE_DIRS-NOTFOUND)
ENDIF()
