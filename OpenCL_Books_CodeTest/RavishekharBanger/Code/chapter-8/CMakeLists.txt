add_executable( chapter8.MatrixMultiplication MatrixMultiplication.cpp )
target_link_libraries( chapter8.MatrixMultiplication ${OPENCL_LIBRARIES} )

add_executable( chapter8.histogram histogram.cpp ../common/bmp_image.cpp)
target_link_libraries( chapter8.histogram ${OPENCL_LIBRARIES} )

add_executable( chapter8.histogram_cu histogram_cu.cpp ../common/bmp_image.cpp)
target_link_libraries( chapter8.histogram_cu ${OPENCL_LIBRARIES} )