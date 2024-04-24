cmake_minimum_required(VERSION 3.24)
find_package(OpenCV REQUIRED)

project(libmodelop)

aux_source_directory(${CMAKE_CURRENT_LIST_DIR} SRC)

add_library(${PROJECT_NAME} ${SRC})
target_include_directories(${PROJECT_NAME} PUBLIC ${ONNXRUNTIME_PATH}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${ONNXRUNTIME_PATH}/lib/onnxruntime.lib ${OpenCV_LIBRARIES})