# PaddleOCR_onnxruntime
使用C++和onnxruntime框架部署PaddleOCR

# 使用方法
修改CMakeLists.txt的ONNXRUNTIME_PATH和OpenCV_DIR，然后编译即可

# 目录结构
.
|-- 1.md
|-- CMakeLists.txt
|-- README.md
|-- resource
|   |-- dicts
|   |   `-- sch_dict.txt
|   `-- models
|       |-- classify.onnx
|       |-- detect.onnx
|       `-- recognize.onnx
`-- source
    |-- Communicate
    |   |-- CMakeLists.txt
    |   |-- server.cpp
    |   `-- server.h
    |-- Loader
    |   |-- CMakeLists.txt
    |   |-- loader.cpp
    |   `-- loader.h
    |-- ModelOP
    |   |-- CMakeLists.txt
    |   |-- clsnetop.cpp
    |   |-- clsnetop.h
    |   |-- dbnetop.cpp
    |   |-- dbnetop.h
    |   |-- recnetop.cpp
    |   `-- recnetop.h
    `-- main.cpp

7 directories, 21 files
