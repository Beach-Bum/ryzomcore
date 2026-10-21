FILE(GLOB SRC *.cpp *.h)

ADD_EXECUTABLE(landscape_probe ${SRC})

TARGET_LINK_LIBRARIES(landscape_probe PRIVATE NeL::pacs NeL::3d NeL::misc)
NL_DEFAULT_PROPS(landscape_probe "NeL, Tools, 3D: landscape_probe")
NL_ADD_RUNTIME_FLAGS(landscape_probe)

INSTALL(TARGETS landscape_probe RUNTIME DESTINATION ${NL_BIN_PREFIX} COMPONENT tools3d)
