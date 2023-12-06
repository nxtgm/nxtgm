option(BUILD_PLUGIN_ILP_HIGHS "Build highs ilp solver" ON)
option(BUILD_PLUGIN_ILP_COIN_CLP "Build coin-clp (i)lp solver" ON)

if(BUILD_PLUGIN_ILP_HIGHS)
  add_subdirectory(highs)
endif()
if(BUILD_PLUGIN_ILP_COIN_CLP)
  add_subdirectory(coin_clp)
endif()
