cmake_minimum_required(VERSION 3.26)

if(UNIX)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif ()

project(CUDAProjectCL LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/nvcc.exe")
find_package(CUDAToolkit )

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY   ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/binaries)

if(UNIX)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/binaries/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/binaries/lib)
elseif(WIN32)
    set(TEST_OUTPUT_PATH ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/binaries)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY   ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/binaries)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/binaries)
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/information.txt
    ${TEST_OUTPUT_PATH}/information.txt
    COPYONLY
)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/music.wav
    ${TEST_OUTPUT_PATH}/music.wav
    COPYONLY
)

set(UNITTEST ON)

if(UNITTEST)
    add_subdirectory(${PROJECT_SOURCE_DIR}/googletests)
    enable_testing()
endif()

add_executable(CUDAProjectCL main.cu)

add_subdirectory(${PROJECT_SOURCE_DIR}/modulation)
target_link_libraries(${PROJECT_NAME} modulation)

add_subdirectory(${PROJECT_SOURCE_DIR}/demodulation)
target_link_libraries(${PROJECT_NAME} demodulation)

add_subdirectory(${PROJECT_SOURCE_DIR}/filtering)
target_link_libraries(${PROJECT_NAME} filtering)

add_subdirectory(${PROJECT_SOURCE_DIR}/myparser)
target_link_libraries(${PROJECT_NAME} myparser)

add_subdirectory(${PROJECT_SOURCE_DIR}/samplefreqchange)
target_link_libraries(${PROJECT_NAME} samplefreqchange)

target_link_libraries(${PROJECT_NAME} CUDA::cudart CUDA::cuda_driver)
target_include_directories(${PROJECT_NAME} PUBLIC  "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include")

