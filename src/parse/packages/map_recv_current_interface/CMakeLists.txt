cmake_minimum_required(VERSION 3.5)
project(map_recv)

set(INTERFACE_VERSION "1.0")
set(COMPONENT_VERSION "1.0")

#--------------------------------------------------

if(${INTERFACE_VERSION})
    message(STATUS "interface_version: ${INTERFACE_VERSION}")
else()
    message(FATAL_ERROR "not set INTERFACE_VERSION")
endif()

if(${COMPONENT_VERSION})
    message(STATUS "component_version: ${COMPONENT_VERSION}")
else()
    message(FATAL_ERROR "not set COMPONENT_VERSION")
endif()

if(USE_DDS)
    add_definitions(-DUSE_DDS)
endif()

if(platform)
    message(STATUS "platform: ${platform}")
else()
    message(FATAL_ERROR "not set platform")
endif()


if(${platform} STREQUAL "821")
    add_definitions(-D_QNX_SOURCE)
    add_definitions(-Dqnx)
endif()

if(${platform} STREQUAL "orin_qnx")
    add_definitions(-D_QNX_SOURCE)
    add_definitions(-Dqnx)
    include_directories($ENV{SDK_ROOT}/include)
    link_directories($ENV{SDK_ROOT}/lib)
endif()

if(${platform} STREQUAL "orin_linux")
    list(APPEND platform_depend_libs pthread dl)
    include_directories(${CMAKE_SYSROOT}/usr/include/iceoryx/v2.90.0)
    message(-----------------$ENV{SDK_ROOT})
endif()

configure_file(
    ${CMAKE_SOURCE_DIR}/src/${PROJECT_NAME}_version.h.in
    ${CMAKE_SOURCE_DIR}/src/${PROJECT_NAME}_version.h
)

add_compile_options(-fPIC -g -std=c++14 -Wall -O2)

set(capilot_DIR /opt/capilot/lib/cmake)
find_package(capilot REQUIRED)
message(STATUS "find capilot: ${capilot_FOUND}")

set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR})

#获取所有cpp文件
file(GLOB_RECURSE SRC  "src/*.cpp")

#获取所有h文件路径
file(GLOB_RECURSE INCLUDE  "src/*.h")
foreach(src ${INCLUDE})
    string(REGEX REPLACE "(.*)/.*h$" "\\1" include_path ${src})
    list(APPEND my_include_dirs ${include_path})
endforeach()
list(REMOVE_DUPLICATES my_include_dirs)


message(STATUS "interface_include_dir: interface${INTERFACE_VERSION}/${platform}/include")
message(STATUS "interface_lib_dir: interface${INTERFACE_VERSION}/${platform}/lib")

include_directories(${my_include_dirs} interface${INTERFACE_VERSION}/${platform}/include)
link_directories(/opt/capilot/lib interface${INTERFACE_VERSION}/${platform}/lib)

# add_executable(test_ ${SRC})
# target_link_libraries(test_ ${capilot_LIBS} drive_interface)


list(REMOVE_ITEM SRC ${CMAKE_SOURCE_DIR}/src/main.cpp)
add_library(${PROJECT_NAME} SHARED ${SRC})
target_link_libraries(${PROJECT_NAME} ${capilot_LIBS} drive_interface)
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${INTERFACE_VERSION})



