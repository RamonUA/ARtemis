# TOP Level Configuration
# ------------------------------------------------------------------------------------------------------
if(CMAKE_GENERATOR MATCHES Xcode AND XCODE_VERSION VERSION_GREATER 4.3)
  cmake_minimum_required(VERSION 2.8.8)
elseif(IOS)
  cmake_minimum_required(VERSION 2.8.0)
else()
  cmake_minimum_required(VERSION 2.6.3)
endif()

# Project name & Configuration
# ------------------------------------------------------------------------------------------------------
SET(PROJ_MAIN_NAME "ARTEMIS")
PROJECT(${PROJ_MAIN_NAME})

# Include Cmake helpers
# ------------------------------------------------------------------------------------------------------
INCLUDE( cmake_tools/scripts/OpenCVUtils.cmake)				# Add OpenCVUtils macros
INCLUDE( cmake_tools/scripts/OpenCVDetectCXXCompiler.cmake)	# Detect compiler and target platform architecture
INCLUDE( cmake_tools/scripts/helperCrosscompile.cmake)		# Helper to crosscompiling and paths
INCLUDE( cmake_tools/scripts/helperFlagsPlatforms.cmake)	# Helper to configure flags for everyplatform
INCLUDE( cmake_tools/scripts/macroProjectInfo.cmake)		# Macro to print basic info
INCLUDE( cmake_tools/scripts/macroProjectAdvancedInfo.cmake)	# Macro to print MAIN info

# Configuration types
SET(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)
IF(DEFINED CMAKE_BUILD_TYPE AND CMAKE_VERSION VERSION_GREATER "2.8")
  SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
ENDIF()


# Configure variables
# ------------------------------------------------------------------------------------------------------

# Paths
SET(LIBRARY_OUTPUT_PATH 		${PROJECT_BINARY_DIR}/bin 				CACHE PATH "Library output" FORCE)
SET(EXECUTABLE_OUTPUT_PATH 		${PROJECT_BINARY_DIR}/bin 				CACHE PATH "Executable output" FORCE)
SET(${PROJ_MAIN_NAME}_PATH_MAIN 	${CMAKE_SOURCE_DIR} 					CACHE PATH "This directory contains initial Path")
SET(${PROJ_MAIN_NAME}_PATH_LIBS 	${LIBRARY_OUTPUT_PATH}					CACHE PATH "This directory contains all libs")
SET(${PROJ_MAIN_NAME}_PATH_EXE 		${PROJECT_BINARY_DIR}					CACHE PATH "This directory contains executables")
#SET(${PROJ_MAIN_NAME}_PATH_3RDPARTY 	${CMAKE_SOURCE_DIR}/modules/3rdparty 				CACHE PATH "This directory contains 3rdparty libraries")
SET(${PROJ_MAIN_NAME}_PATH_DATA 	${${PROJ_MAIN_NAME}_PATH_MAIN}/extras/data/ 		CACHE PATH "This directory contains all DATA and RESOURCES")
SET(${PROJ_MAIN_NAME}_PATH_INSTALL 	${PROJECT_BINARY_DIR}/install/${PROJ_MAIN_NAME} 	CACHE PATH "This directory to install prebuilt")

# Build type
IF(TARGET_PLATFORM STREQUAL TARGET_ANDROID)
  SET(${PROJ_MAIN_NAME}_LIB_TYPE "STATIC" CACHE STRING "Choose STATIC or SHARED to generate libraries" FORCE)
ELSEIF(TARGET_PLATFORM STREQUAL TARGET_WIN32)
  SET(${PROJ_MAIN_NAME}_LIB_TYPE "SHARED" CACHE STRING "Choose STATIC or SHARED to generate libraries" FORCE)
ELSEIF(TARGET_PLATFORM STREQUAL TARGET_LINUX)
#SET(${PROJ_MAIN_NAME}_LIB_TYPE "STATIC" CACHE STRING "Choose STATIC or SHARED to generate libraries" FORCE)
  #SET(${PROJ_MAIN_NAME}_LIB_TYPE "SHARED" CACHE STRING "Choose STATIC or SHARED to generate libraries" FORCE)
ELSEIF(TARGET_PLATFORM STREQUAL TARGET_APPLE)
  SET(${PROJ_MAIN_NAME}_LIB_TYPE "SHARED" CACHE STRING "Choose STATIC or SHARED to generate libraries" FORCE)
ELSEIF(TARGET_PLATFORM STREQUAL TARGET_IOS)
  SET(${PROJ_MAIN_NAME}_LIB_TYPE "STATIC" CACHE STRING "Choose STATIC or SHARED to generate libraries" FORCE)
ELSE()
   MESSAGE(FATAL_ERROR "ERROR ${PROJ_MAIN_NAME}_LIB_TYPE")
ENDIF()

# Configure options
# ------------------------------------------------------------------------------------------------------
OCV_OPTION(CMAKE_VERBOSE "Verbose mode" ON )
OCV_OPTION(${PROJ_MAIN_NAME}_BUILD_PROFILE       "Add flags to profiling UNIX" 					on )
OCV_OPTION(${PROJ_MAIN_NAME}_BUILD_CPP11         "Force compile with standard C++11" 				on )
OCV_OPTION(${PROJ_MAIN_NAME}_BUILD_SAMPLE       "Compile the app that is used to calibrate the system" 					OFF )

# Configuration per platform
# ------------------------------------------------------------------------------------------------------
include( cmake_tools/scripts/helperFlagsPlatforms.cmake)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${${PROJ_MAIN_NAME}_PATH_LIBS})
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${${PROJ_MAIN_NAME}_PATH_LIBS})

# Configuration current project
# ------------------------------------------------------------------------------------------------------
SET(CMAKE_DEBUG_POSTFIX "d")

# Dependencies
# ------------------------------------------------------------------------------------------------------

IF(TARGET_PLATFORM STREQUAL TARGET_WIN32)
  SET(OpenCV_STATIC OFF)			# force opencv to link dynamic
ENDIF(TARGET_PLATFORM STREQUAL TARGET_WIN32)
FIND_PACKAGE(OpenCV REQUIRED)
IF(OpenCV_DIR STREQUAL NOT_FOUND)
  MESSAGE(FATAL_ERROR "Install OpenCV")
ENDIF()
include_directories(${OpenCV_DIR}/include)
#MESSAGE(${OpenCV_LIBS})
#list(REMOVE_ITEM OpenCV_LIBS ippicv.lib)

IF( (TARGET_PLATFORM STREQUAL TARGET_ANDROID) AND (${OpenCV_DIR} STREQUAL "/usr/share/OpenCV"))
  MESSAGE("--***-- Warning: Select OpenCV_Dir & OpenCV_CONFIG_PATH manually to OpenCV/sdk/native/jni")
ENDIF()

IF(TARGET_PLATFORM STREQUAL TARGET_ANDROID OR TARGET_PLATFORM STREQUAL TARGET_IOS)
	ADD_DEFINITIONS(-DNOGLUT)	# Temporal avoid use glut in Android & ios
ENDIF()

#find_package(PCL 1.2 REQUIRED)
#include_directories(${PCL_INCLUDE_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})


# Project Logic
# ------------------------------------------------------------------------------------------------------
IF(NOT TARGET_PLATFORM STREQUAL TARGET_NONE)
  ######################################################################################
  # Logic configuration
  ######################################################################################
  IF(${PROJ_MAIN_NAME}_BUILD_CPP11)
    MESSAGE("Adding C++11 flags, NEEDED to be supported by compiler")
    SET(CMAKE_C_FLAGS "-std=c11 ${CMAKE_C_FLAGS} ")
    SET(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} ")
  ELSEIF(TARGET_PLATFORM STREQUAL TARGET_LINUX)
  		MESSAGE("Adding C++110X flags")
		SET(CMAKE_C_FLAGS "-std=c++0x ${CMAKE_C_FLAGS} ")
		SET(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS} ")
  ENDIF()

  SET(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} ")

  ######################################################################################
  # Print info
  ######################################################################################
  PRINTADVANCEDINFO(${PROJ_MAIN_NAME})

  ######################################################################################
  # ADD SUB PROJECTS
  ######################################################################################
  ADD_SUBDIRECTORY(ARtemis)

  IF(${PROJ_MAIN_NAME}_BUILD_SAMPLE)
	SET(JBCTRACK_SAMPLE_PATH_DATA ${CMAKE_SOURCE_DIR}/extras/data CACHE PATH "This directory contains all DATA")
	ADD_SUBDIRECTORY(samples)
  ENDIF()

ELSE()
  message(FATAL_ERROR " TARGET no autodetected. Please, select a TARGET manually. ")
ENDIF()

LIST(APPEND ${PROJ_MAIN_NAME}_LIBS ARtemis)
LIST(APPEND ${PROJ_MAIN_NAME}_LIBS_DEBUG ARtemis)

configure_file(ARtemisConfig.cmake.in
  "${PROJECT_SOURCE_DIR}/ARtemisConfig.cmake")
