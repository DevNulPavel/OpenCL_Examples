add_executable( chapter11.linear_regression linear_regression.cpp )
target_link_libraries( chapter11.linear_regression ${OPENCL_LIBRARIES} )

add_executable( chapter11.parabolic_regression parabolic_regression.cpp )
target_link_libraries( chapter11.parabolic_regression ${OPENCL_LIBRARIES} )

add_executable( chapter11.bitonic_sort bitonic_sort.cpp )
target_link_libraries( chapter11.bitonic_sort ${OPENCL_LIBRARIES} )

add_executable( chapter11.knn_classification knn_classification.cpp )
target_link_libraries( chapter11.knn_classification ${OPENCL_LIBRARIES} )
