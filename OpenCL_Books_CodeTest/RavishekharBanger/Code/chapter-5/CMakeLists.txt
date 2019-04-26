add_executable( chapter5.create_binary create_binary.cpp )
target_link_libraries( chapter5.create_binary ${OPENCL_LIBRARIES} )

add_executable( chapter5.saxpy_binary saxpy_binary.cpp )
target_link_libraries( chapter5.saxpy_binary ${OPENCL_LIBRARIES} )

