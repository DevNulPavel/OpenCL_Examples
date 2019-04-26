
set( JPEGDecoder.Source Framewave.cpp jpeg_marker.cpp image.cpp JPEG_Decoder.cpp scan.cpp)
set( JPEGDecoder.Headers jpeg_marker.h Framewave.h image.h JPEG_Decoder.hpp JPEG_Decoder_Kernels.h )
set( JPEGDecoder.Files ${JPEGDecoder.Source} ${JPEGDecoder.Headers} )

add_executable( chapter9.JPEGDecoder ${JPEGDecoder.Files} )
target_link_libraries( chapter9.JPEGDecoder ${OPENCL_LIBRARIES} )

