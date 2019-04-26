# This file is used obtained from http://gitorious.org/findopencl 
# - Try to find OpenCL
# This module tries to find an OpenCL implementation on your system. It supports
# AMD, Intel and NVIDIA implementations, but should work, too.
# Prerequisites
# Atleast one of the OpenCL SDKs should be installed. 
# --If AMD APPSDK is installed then make sure the environment AMDAPPSDKROOT variable is set
# --If NVDIA GPU Compute SDK is installed then make sure the environment NVSDKCOMPUTE_ROOT variable is set
# --If Intel OpenCL Application SDK is installed then make sure the environment INTELOCLSDKROOT variable is set
#
# Once done this will define
#  OPENCL_FOUND        - system has OpenCL
#  OPENCL_INCLUDE_DIRS  - the OpenCL include directory
#  OPENCL_LIBRARIES    - link these to use OpenCL
#
# 

FIND_PACKAGE(PackageHandleStandardArgs)

SET (OPENCL_VERSION_STRING "1.2.0")
SET (OPENCL_VERSION_MAJOR 1)
SET (OPENCL_VERSION_MINOR 2)
SET (OPENCL_VERSION_PATCH 0)

option( BUILD_64 "Linux only 64 bit build" ON)        
option( BUILD_AMD_OPENCL   "Create Build for AMD OpenCL implementation" ON )
option( BUILD_NV_OPENCL    "Create Build for NV OpenCL implementation" OFF )
option( BUILD_INTEL_OPENCL "Create Build for INTEL OpenCL implementation" OFF )
option( BUILD_APPLE_OPENCL "Create Build for APPLE OpenCL implementation" OFF )

if( BUILD_64 )
    set_property( GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE )
    message( STATUS "64bit build - FIND_LIBRARY_USE_LIB64_PATHS TRUE" )
else()
    set_property( GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS FALSE )
    message( STATUS "32bit build - FIND_LIBRARY_USE_LIB64_PATHS FALSE" )
endif()


IF(${BUILD_AMD_OPENCL} STREQUAL ON)
    IF ( DEFINED ENV{AMDAPPSDKROOT} ) 
        message (STATUS "AMDAPPSDKROOT environment variable is set")
    ELSE (  ) 
        message ("AMDAPPSDKROOT is not set or APP SDK is not installed")
    ENDIF(  )
ENDIF (${BUILD_AMD_OPENCL} STREQUAL ON)

IF(${BUILD_NV_OPENCL} STREQUAL ON)
    IF ( DEFINED ENV{NVSDKCOMPUTE_ROOT} ) 
        message (STATUS "NVSDKCOMPUTE_ROOT is set")
    ELSE ( ) 
        message ("NVSDKCOMPUTE_ROOT is not set or NVIDIA GPU compute SDK is not installed")
    ENDIF( )
ENDIF ( )

IF(${BUILD_INTEL_OPENCL} STREQUAL ON)
    IF (DEFINED ENV{INTELOCLSDKROOT}) 
        message (STATUS "INTELOCLSDKROOT is set")
    ELSE ( ) 
        message ("INTELOCLSDKROOT is not set. Install INTEL OpenCL Application or set the path manually")
    ENDIF( )
ENDIF ( )


IF (APPLE)

    FIND_LIBRARY(OPENCL_LIBRARIES OpenCL DOC "OpenCL lib for OSX")
    FIND_PATH(OPENCL_INCLUDE_DIRS OpenCL/cl.h DOC "Include for OpenCL on OSX")
    FIND_PATH(_OPENCL_CPP_INCLUDE_DIRS OpenCL/cl.hpp DOC "Include for OpenCL CPP bindings on OSX")

ELSEIF (WIN32)

        # The AMD SDK currently installs both x86 and x86_64 libraries
        IF(${BUILD_AMD_OPENCL} STREQUAL ON)
                #http://developer.amd.com/download/AMD_APP_SDK_Installation_Notes.pdf
                IF( ${CMAKE_CL_64} EQUAL 1 )     # using Visual Studio Win64 Generator
                        SET(OPENCL_LIB_DIR "$ENV{AMDAPPSDKROOT}/lib/x86_64")
                ELSE ( ${CMAKE_CL_64} EQUAL 1 )  # using Visual Studio Generator - non Win64
                        SET(OPENCL_LIB_DIR "$ENV{AMDAPPSDKROOT}/lib/x86")
                ENDIF( ${CMAKE_CL_64} EQUAL 1 )

                FIND_LIBRARY(OPENCL_LIBRARIES OpenCL.lib PATHS ${OPENCL_LIB_DIR} )
                GET_FILENAME_COMPONENT(_OPENCL_INC_CAND ${OPENCL_LIB_DIR}/../../include ABSOLUTE)
                # On Win32 search relative to the library
                FIND_PATH(OPENCL_INCLUDE_DIRS CL/cl.h PATHS "${_OPENCL_INC_CAND}" )
                FIND_PATH(_OPENCL_CPP_INCLUDE_DIRS CL/cl.hpp PATHS "${_OPENCL_INC_CAND}" ENV OpenCL_INCPATH)


        ELSEIF ( ${BUILD_NV_OPENCL} STREQUAL ON ) 
                #http://developer.download.nvidia.com/compute/cuda/3_1/sdk/docs/OpenCL_release_notes.txt
                IF( ${CMAKE_CL_64} EQUAL 1 )     # using Visual Studio Win64 Generator
                        SET(OPENCL_LIB_DIR "$ENV{NVSDKCOMPUTE_ROOT}/OpenCL/common/lib/x64")
                ELSE ( ${CMAKE_CL_64} EQUAL 1 )  # using Visual Studio Generator - non Win64
                        SET(OPENCL_LIB_DIR "$ENV{NVSDKCOMPUTE_ROOT}/OpenCL/common/lib/win32")
                ENDIF( ${CMAKE_CL_64} EQUAL 1 )

                FIND_LIBRARY(OPENCL_LIBRARIES OpenCL.lib PATHS ${OPENCL_LIB_DIR} )
                GET_FILENAME_COMPONENT(_OPENCL_INC_CAND ${OPENCL_LIB_DIR}/../../include ABSOLUTE)
                # On Win32 search relative to the library
                FIND_PATH(OPENCL_INCLUDE_DIRS CL/cl.h PATHS "${_OPENCL_INC_CAND}" )
                FIND_PATH(_OPENCL_CPP_INCLUDE_DIRS CL/cl.hpp PATHS "${_OPENCL_INC_CAND}" ENV OpenCL_INCPATH)
                
                
        ELSE ( ${BUILD_AMD_OPENCL} STREQUAL ON ) 
                #http://software.intel.com/en-us/articles/intel-sdk-for-opencl-applications-2013-release-notes
                IF( ${CMAKE_CL_64} EQUAL 1 )     # using Visual Studio Win64 Generator
                        SET(OPENCL_LIB_DIR "$ENV{INTELOCLSDKROOT}/lib/x64")
                ELSE ( ${CMAKE_CL_64} EQUAL 1 )  # using Visual Studio Generator - non Win64
                        SET(OPENCL_LIB_DIR "$ENV{INTELOCLSDKROOT}/lib/x86")
                ENDIF( ${CMAKE_CL_64} EQUAL 1 )

                FIND_LIBRARY(OPENCL_LIBRARIES OpenCL.lib PATHS ${OPENCL_LIB_DIR} )
                GET_FILENAME_COMPONENT(_OPENCL_INC_CAND ${OPENCL_LIB_DIR}/../../include ABSOLUTE)
                # On Win32 search relative to the library
                FIND_PATH(OPENCL_INCLUDE_DIRS CL/cl.h PATHS "${_OPENCL_INC_CAND}" )
                FIND_PATH(_OPENCL_CPP_INCLUDE_DIRS CL/cl.hpp PATHS "${_OPENCL_INC_CAND}" ENV OpenCL_INCPATH)

        ELSE ( ${BUILD_AMD_OPENCL} STREQUAL ON ) 
                message("NO OpenCL Build system is selected.")
        ENDIF( ${BUILD_AMD_OPENCL} STREQUAL ON)        


ELSE (APPLE)

        # Unix style platforms
    get_property( LIB64 GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS )

        IF(${BUILD_AMD_OPENCL} STREQUAL ON)
            SET(OPENCL_ROOT "$ENV{AMDAPPSDKROOT}/")
        ELSEIF ( ${BUILD_NV_OPENCL} STREQUAL ON ) 
            SET(OPENCL_ROOT "$ENV{NVSDKCOMPUTE_ROOT}/")
        ELSE ( ${BUILD_AMD_OPENCL} STREQUAL ON ) 
            SET(OPENCL_ROOT "$ENV{INTELOCLSDKROOT}/")
        ENDIF( ${BUILD_AMD_OPENCL} STREQUAL ON)
        IF( LIB64 )
        message("building 64-bit lib")
        FIND_LIBRARY( OPENCL_LIBRARIES
                          OpenCL
                      HINTS
                          ${OPENCL_ROOT}/lib
                          $ENV{OPENCL_ROOT}/lib
                      DOC "OpenCL dynamic library path"
                      PATH_SUFFIXES x86_64 x64
        )
        ELSE( )
        message("building 32-bit lib")
        FIND_LIBRARY( OPENCL_LIBRARIES
                          OpenCL
                      HINTS
                          ${OPENCL_ROOT}/lib
                          $ENV{OPENCL_ROOT}/lib
                      DOC "OpenCL dynamic library path"
                      PATH_SUFFIXES x86
                    )
        ENDIF( )


        #FIND_LIBRARY(OPENCL_LIBRARIES OpenCL
        #        PATHS ENV LD_LIBRARY_PATH ENV OpenCL_LIBPATH
        #)

        GET_FILENAME_COMPONENT(OPENCL_LIB_DIR ${OPENCL_LIBRARIES} PATH)
        GET_FILENAME_COMPONENT(_OPENCL_INC_CAND ${OPENCL_LIB_DIR}/../../include ABSOLUTE)

        # The AMD SDK currently does not place its headers
        # in /usr/include, therefore also search relative
        # to the library
    IF (${BUILD_AMD_OPENCL} STREQUAL ON)
            SET(OCL_INSTALL_PATH "/opt/AMDAPP/include")
    ELSEIF (${BUILD_AMD_OPENCL} STREQUAL ON)
            SET(OCL_INSTALL_PATH "/usr/local/cuda/include")
        ENDIF ()
        FIND_PATH(OPENCL_INCLUDE_DIRS CL/cl.h PATHS ${_OPENCL_INC_CAND} ${OCL_INSTALL_PATH} ENV OpenCL_INCPATH)
        FIND_PATH(_OPENCL_CPP_INCLUDE_DIRS CL/cl.hpp PATHS ${_OPENCL_INC_CAND} ${OCL_INSTALL_PATH} ENV OpenCL_INCPATH)

ENDIF (APPLE)



FIND_PACKAGE_HANDLE_STANDARD_ARGS(OpenCL DEFAULT_MSG OPENCL_LIBRARIES OPENCL_INCLUDE_DIRS)

IF(_OPENCL_CPP_INCLUDE_DIRS)
    SET( OPENCL_HAS_CPP_BINDINGS TRUE )
    LIST( APPEND OPENCL_INCLUDE_DIRS ${_OPENCL_CPP_INCLUDE_DIRS} )
    # This is often the same, so clean up
    LIST( REMOVE_DUPLICATES OPENCL_INCLUDE_DIRS )
ENDIF(_OPENCL_CPP_INCLUDE_DIRS)

MARK_AS_ADVANCED(
  OPENCL_INCLUDE_DIRS
)
message (STATUS "OPENCL_FOUND :" ${OPENCL_FOUND} )
message (STATUS "OPENCL_INCLUDE_DIRS :" ${OPENCL_INCLUDE_DIRS} )
message (STATUS "OPENCL_LIBRARIES :" ${OPENCL_LIBRARIES} )
