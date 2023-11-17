option(BUILD_PLUGIN_ILP_HIGHS "Build highs ilp solver" ON)

if(BUILD_PLUGIN_ILP_HIGHS)
  add_subdirectory(highs)
endif()
