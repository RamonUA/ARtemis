SET(PROJ_NAME ARtemis)
PROJECT(${PROJ_NAME})

STATUS(" + Adding module ${PROJ_NAME} ")

# Include directories with headers
#---------------------------------------------------#
INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include )

# Source
#---------------------------------------------------#
FILE(GLOB ${PROJ_NAME}_SRC
    "src/*.cpp"
)
FILE(GLOB ${PROJ_NAME}_HEADERS
    "include/*.h"
)

# Create Library
#---------------------------------------------------#
ADD_LIBRARY(${PROJ_NAME} ${${PROJ_MAIN_NAME}_LIB_TYPE} ${${PROJ_NAME}_HEADERS} ${${PROJ_NAME}_SRC} ) 

# Link libraries with Project
#---------------------------------------------------#
LINK_DIRECTORIES( ${${PROJ_MAIN_NAME}_PATH_LIBS} )

TARGET_LINK_LIBRARIES(${PROJ_NAME} ${OpenCV_LIBS})
#TARGET_LINK_LIBRARIES(${PROJ_NAME} ${PCL_LIBRARIES})

# Install
#---------------------------------------------------#
INSTALL(TARGETS ${PROJ_NAME}
        DESTINATION "${${PROJ_MAIN_NAME}_PATH_INSTALL}/${PROJ_NAME}/lib/${CMAKE_BUILD_TYPE}/"
        )
INSTALL(FILES ${${PROJ_NAME}_HEADERS}
        DESTINATION "${${PROJ_MAIN_NAME}_PATH_INSTALL}/${PROJ_NAME}/include"
        )

#PRINTBASICINFO(${PROJ_NAME})
STATUS(" - Module ${PROJ_NAME} added!")
