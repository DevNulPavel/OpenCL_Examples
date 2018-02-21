set(PROJECT_NAME "OpenCL")
set(EXECUTABLE_NAME "OpenCL")

####################################################
# Настройка CMake сборки
####################################################
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
cmake_policy(VERSION 3.0)
if (NOT CMAKE_VERSION VERSION_LESS "3.1")
	cmake_policy(SET CMP0054 NEW)
endif()

# Информация о текущих настройках
function(print_current_cmake_config)
    # https://cmake.org/Wiki/CMake_Useful_Variables
    message("******** Project compilation settings ***********")
    message("System = ${CMAKE_SYSTEM}")
    message("System name = ${CMAKE_SYSTEM_NAME}")
    message("System version = ${CMAKE_SYSTEM_VERSION}")
    message("CMake Generator = ${CMAKE_GENERATOR}")
    message("CMake build type = ${CMAKE_BUILD_TYPE}")
    message("CC compiler = ${CMAKE_C_COMPILER}")
    message("CC Flags (Debug) = ${CMAKE_C_FLAGS_DEBUG}")
    message("CC Flags (Release) = ${CMAKE_C_FLAGS_RELEASE}")
    message("CXX Flags (Debug) = ${CMAKE_CXX_FLAGS_DEBUG}")
    message("CXX Flags (Release) = ${CMAKE_CXX_FLAGS_RELEASE}")
endfunction()

# Цвета
option(CMAKE_COLOR_MAKEFILE "Color makefile" ON)

# Оставляем толкьо релизную и дебажную сборку
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)

# Инициализация проекта
project(${PROJECT_NAME})

enable_testing()


####################################################
# Библиотеки в системе
####################################################
message("******** System libs search ***********")

find_package(OpenCL)
if(OpenCL_FOUND)
    include_directories(${OpenCL_INCLUDE_DIRS})
    set(LIBS_LINK_NAMES ${LIBS_LINK_NAMES} ${OpenCL_LIBRARIES})    
else()
    message(FATAL_ERROR "OpenCL lib not found!")
endif()

# Системные библиотеки
if (UNIX AND NOT APPLE)
	set(SYSTEM_LIBRARIES pthread m)
endif()


####################################################
# Настройка сборки текущего проекта
####################################################
# Сразу отключаем ненужные предупреждения на Windows
if (MSYS)
	add_definitions ( -D_CRT_SECURE_NO_WARNINGS )
	add_definitions ( -D_SCL_SECURE_NO_WARNINGS )
endif ()

# Не кастомные флаги компиляции
if(APPLE)
	set(PROJECT_LINK_FLAGS "-stdlib=libc++")
	set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -O0 -Wall -g")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -O2 -Wall")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -Wall -g -stdlib=libc++ -std=c++11")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O2 -Wall -stdlib=libc++ -std=c++11")
endif ()
if (UNIX AND NOT APPLE)
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -O0 -Wall -g")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -O2 -Wall")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -Wall -g -std=c++11")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O2 -Wall -std=c++11")
endif ()
if (MSYS)
	# микрософту не нужен флаг -std=c++11
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} /O0 /Wall /g")
    set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} /O2 /Wall")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} /O0 /Wall /g")
    set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} /O2 /Wall")
endif ()
if (WIN32)
    # микрософту не нужен флаг -std=c++11 +  -stdlib=libc++??
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -O0 -Wall -g -std=c99")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -O2 -Wall -std=c99")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -Wall -g -std=c++11")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O2 -Wall -std=c++11")
endif ()

# Вывод информации о настройке копиляции (Функция в CMake/Helpers.cmake)
print_current_cmake_config()


####################################################
# Исходники
####################################################
set(ALL_SOURCES CL.hpp main.cpp MathCode.h MathCode.cpp)
set(EDIT_SOURCES Kernel.cl CL.hpp main.cpp MathCode.h MathCode.cpp)	# Чтобы отображалось в исходниках
source_group("Sources" FILES ${EDIT_SOURCES})

# тип сборки
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Release" FORCE)
endif()


####################################################
# Генерация
####################################################
message("******** Generating started ***********")
# Из найденных исходников выставляем генерацию бинарника
add_executable(${EXECUTABLE_NAME} ${ALL_SOURCES})

# Конкретные пути к бинарнику
set_target_properties( ${EXECUTABLE_NAME}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}
)

# Создаем символьную ссылку на файлик .cl
if (WIN32)
    string(REGEX REPLACE "/" "\\\\" SOURCE_RES_PATH "${CMAKE_SOURCE_DIR}/Kernel.cl")
    string(REGEX REPLACE "/" "\\\\" RESULT_RES_PATH "${CMAKE_BINARY_DIR}")
	string(REGEX REPLACE "/" "\\\\" RESULT_FILE_PATH "${CMAKE_BINARY_DIR}/Kernel.cl")
    message("COPY Kernel.cl COMMAND: xcopy ${SOURCE_RES_PATH} ${RESULT_RES_PATH} /y")
    add_custom_command (
        TARGET ${EXECUTABLE_NAME}
        POST_BUILD
		COMMAND del /F "${RESULT_FILE_PATH}"
        COMMAND xcopy "${SOURCE_RES_PATH}" "${RESULT_RES_PATH}"
    )
else()
    add_custom_command (
        TARGET ${EXECUTABLE_NAME}
        POST_BUILD
        COMMAND rm -f "${CMAKE_BINARY_DIR}/Kernel.cl"
        COMMAND ln -s "${CMAKE_SOURCE_DIR}/Kernel.cl" "${CMAKE_BINARY_DIR}/Kernel.cl"
    )
endif()


# Библиотеки, с которыми будем ликовать
# link_directories(${LIBS_LINK_FOLDERS})
target_link_libraries (${EXECUTABLE_NAME} ${LIBS_LINK_NAMES} ${SYSTEM_LIBRARIES})


