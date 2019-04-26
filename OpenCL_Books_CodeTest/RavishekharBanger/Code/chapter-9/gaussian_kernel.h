#ifndef OCL_GAUSSIAN_KERNEL_H
#define OCL_GAUSSIAN_KERNEL_H
static const char *gaussian_kernel =
"                                                                                                      \n"
"__constant sampler_t image_sampler = CLK_NORMALIZED_COORDS_FALSE                                      \n"
"                                     | CLK_ADDRESS_CLAMP_TO_EDGE                                      \n"
"                                     | CLK_FILTER_NEAREST;                                            \n"
"                                                                                                      \n"
"__kernel void gaussian_filter_kernel(__read_only image2d_t iimage,                                    \n"
"                                     __write_only image2d_t oimage,                                   \n"
"                                     __constant float *filter, int windowSize)                        \n"
"{                                                                                                     \n"
"    unsigned int x = get_global_id(0);                                                                \n"
"    unsigned int y = get_global_id(1);                                                                \n"
"    int halfWindow = windowSize/2;                                                                    \n"
"    float4 pixelValue;                                                                                \n"
"    float4 computedFilter=0.0f;                                                                       \n"
"    int i, j, ifilter, jfilter;                                                                       \n"
"                                                                                                      \n"
"    for(i=-halfWindow, ifilter=0; i<=halfWindow; i++, ifilter++){                                     \n"
"       for(j=-halfWindow, jfilter=0; j<=halfWindow; j++, jfilter++){                                  \n"
"           pixelValue = read_imagef(iimage, image_sampler,                                            \n"
"                                    (int2)(x+i, y+j));                                                \n"
"           computedFilter += filter[ifilter*windowSize+jfilter]*pixelValue;                           \n"
"       }                                                                                              \n"
"    }                                                                                                 \n"
"                                                                                                      \n"
"                                                                                                      \n"
"    write_imagef(oimage, (int2)(x, y), computedFilter);                                               \n"
"}                                                                                                     \n"
"                                                                                                      \n"
"                                                                                                      \n";

#endif