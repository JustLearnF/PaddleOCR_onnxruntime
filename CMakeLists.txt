cmake_minimum_required(VERSION 3.24)

macro(GLOBAL_SET ARG1 ARG2)
set(${ARG1} ${ARG2} CACHE STRING "" FORCE)
endmacro()

GLOBAL_SET(CMAKE_CXX_STANDARD 17)

GLOBAL_SET(ONNXRUNTIME_PATH "D:/Tools/DeepLearning/DeepLearningFrame/onnxruntime-1.17.1")
GLOBAL_SET(OpenCV_DIR "D:/Tools/OpenCV/build")

SET(MODULEDIR ${CMAKE_SOURCE_DIR}/source)

project(OCR LANGUAGES CXX)
add_compile_options("/utf-8")

add_subdirectory(${MODULEDIR}/Loader)
add_subdirectory(${MODULEDIR}/ModelOP)
add_subdirectory(${MODULEDIR}/Communicate)

aux_source_directory(${CMAKE_SOURCE_DIR}/source SRC)
include_directories(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/source)

add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME}  libloader libmodelop libcommunicate)