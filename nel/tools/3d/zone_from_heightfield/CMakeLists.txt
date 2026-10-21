FILE(GLOB SRC *.cpp *.h)

ADD_EXECUTABLE(zone_from_heightfield ${SRC})

TARGET_LINK_LIBRARIES(zone_from_heightfield PRIVATE NeL::3d NeL::misc)
NL_DEFAULT_PROPS(zone_from_heightfield "NeL, Tools, 3D: zone_from_heightfield")
NL_ADD_RUNTIME_FLAGS(zone_from_heightfield)

INSTALL(TARGETS zone_from_heightfield RUNTIME DESTINATION ${NL_BIN_PREFIX} COMPONENT tools3d)
