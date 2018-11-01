SET(PROJ_NAME sample_ARtemis)
PROJECT(${PROJ_NAME})

STATUS(" + Adding sample ${PROJ_NAME} ")

# Configure file
configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/include/ARtemisSampleConfig.h.in"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/ARtemisSampleConfig.h" )
				
# Include directories with headers
#---------------------------------------------------#
INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include )
INCLUDE_DIRECTORIES( ${${PROJ_MAIN_NAME}_PATH_MAIN}/ARtemis/include )

# Source
#---------------------------------------------------#
FILE(GLOB ${PROJ_NAME}_SRC
    "src/*.cpp"
)
FILE(GLOB ${PROJ_NAME}_HEADERS
    "include/*.h"
)

# Create Executable
#---------------------------------------------------#
ADD_EXECUTABLE( ${PROJ_NAME} ${${PROJ_NAME}_HEADERS} ${${PROJ_NAME}_SRC} )

# Link libraries with Project
#---------------------------------------------------#
LINK_DIRECTORIES( ${${PROJ_MAIN_NAME}_PATH_LIBS} )

TARGET_LINK_LIBRARIES(${PROJ_NAME} ${OpenCV_LIBS} )
TARGET_LINK_LIBRARIES(${PROJ_NAME} ARtemis )

#PRINTBASICINFO(${PROJ_NAME})
STATUS(" - ${PROJ_NAME} added!")