cmake_minimum_required(VERSION 3.24)

project(libloader)

aux_source_directory(${CMAKE_CURRENT_LIST_DIR} SRC)

add_library(${PROJECT_NAME} ${SRC})
target_include_directories(${PROJECT_NAME} PUBLIC)