set(chapter1.saxpy.Source saxpy.cpp)

add_executable( chapter1.saxpy ${chapter1.saxpy.Source} )
target_link_libraries( chapter1.saxpy ${OPENCL_LIBRARIES} )

set_property( TARGET chapter1.saxpy PROPERTY FOLDER "chapter1/")
