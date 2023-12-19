
option(BUILD_PLUGIN_PROPOSAL_GEN_ALPHA_EXPANSION "Build alpha expansion proposal generator plugin"  ON)
option(BUILD_PLUGIN_PROPOSAL_GEN_OPTIMIZER_BASED "Build optimizer based proposal generator plugin"  ON)
option(BUILD_PLUGIN_PROPOSAL_MATCHMAKER          "Build matchmaker proposal generator plugin"       ON)
option(BUILD_PLUGIN_PROPOSAL_RANDOM              "Build random proposal generator plugin"           ON)
option(BUILD_PLUGIN_PROPOSAL_MATCHING_BP         "Build proposal generator based on matching bp"    ON)
option(BUILD_PLUGIN_PROPOSAL_GEN_TESTING         "Build proposal generator plugin used for testing" ON)



if(BUILD_PLUGIN_PROPOSAL_GEN_ALPHA_EXPANSION)
  add_subdirectory(alpha_expansion)
endif()

if(BUILD_PLUGIN_PROPOSAL_GEN_OPTIMIZER_BASED)
  add_subdirectory(optimizer_based)
endif()

if(BUILD_PLUGIN_PROPOSAL_MATCHMAKER)
  add_subdirectory(matchmaker)
endif()

if(BUILD_PLUGIN_PROPOSAL_RANDOM)
  add_subdirectory(random)
endif()

if(BUILD_PLUGIN_PROPOSAL_MATCHING_BP)
  add_subdirectory(matching_bp)
endif()

if(BUILD_PLUGIN_PROPOSAL_GEN_TESTING)
  add_subdirectory(testing)
endif()
