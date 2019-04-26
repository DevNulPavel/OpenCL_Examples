# This is the root cmake file. This one shall be used to create the
# different project build related files

cmake_minimum_required( VERSION 2.8 )
project( OpenCL_By_Example )

set( CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR} )
if ( MSVC_IDE )
    set( CMAKE_CXX_FLAGS "-DWIN32" )
    set( CMAKE_C_FLAGS "-DWIN32" )
endif( )
find_package( OpenCL REQUIRED )
include_directories( ${OPENCL_INCLUDE_DIRS} include )
SUBDIRS( chapter-1/
         chapter-2/
         chapter-3/
         chapter-4/
         chapter-5/
         chapter-6/
         chapter-7/
         chapter-8/
         chapter-9/
         chapter-11/
         )
