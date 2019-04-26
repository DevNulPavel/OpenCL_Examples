add_executable( chapter3.histogram histogram.cpp ../common/bmp_image.cpp)
target_link_libraries( chapter3.histogram ${OPENCL_LIBRARIES} )

add_executable( chapter3.bufferRectangularReads bufferRectangularReads.cpp )
target_link_libraries( chapter3.bufferRectangularReads ${OPENCL_LIBRARIES} )

add_executable( chapter3.copyRectangular copyRectangular.cpp )
target_link_libraries( chapter3.copyRectangular ${OPENCL_LIBRARIES} )

