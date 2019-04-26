include_directories( "C:/Users/rbanger/Documents/AMD APP/samples/opencl/SDKUtil/include" )

if(NOT MSVC_IDE )
    set ( RUNTIME_LIBRARY -lrt )
endif()
set( gaussian.Source ../common/bmp_image.cpp gaussian.cpp ../common/Timer.cpp)
set( gaussian.Headers ../include/bmp_image.h gaussian_kernel.h gaussian.h ../include/Timer.h )
set( gaussian.Files ${gaussian.Source} ${gaussian.Headers} )
add_executable( chapter9.gaussian  ${gaussian.Files} )
target_link_libraries( chapter9.gaussian ${OPENCL_LIBRARIES} ${RUNTIME_LIBRARY})

set( sobel.Source ../common/bmp_image.cpp sobel.cpp ../common/Timer.cpp)
set( sobel.Headers ../include/bmp_image.h sobel_kernel.h sobel.h ../include/Timer.h )
set( sobel.Files ${sobel.Source} ${sobel.Headers} )
add_executable( chapter9.sobel  ${sobel.Files} )
target_link_libraries( chapter9.sobel ${OPENCL_LIBRARIES} ${RUNTIME_LIBRARY})

set( median.Source ../common/bmp_image.cpp median.cpp ../common/Timer.cpp)
set( median.Headers ../include/bmp_image.h median_kernel.h median.h ../include/Timer.h )
set( median.Files ${median.Source} ${median.Headers} )
add_executable( chapter9.median  ${median.Files} )
target_link_libraries( chapter9.median ${OPENCL_LIBRARIES} ${RUNTIME_LIBRARY})

SUBDIRS( jpeg_decoder )