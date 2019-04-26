
add_executable( chapter2.get_device_property get_device_property.cpp )
target_link_libraries( chapter2.get_device_property ${OPENCL_LIBRARIES} )

add_executable( chapter2.get_platform_property get_platform_property.cpp )
target_link_libraries( chapter2.get_platform_property ${OPENCL_LIBRARIES} )