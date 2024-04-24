# PaddleOCR_onnxruntime
使用C++和onnxruntime框架部署PaddleOCR

# 使用方法
修改CMakeLists.txt的ONNXRUNTIME_PATH和OpenCV_DIR，然后编译即可

# 目录结构

|-- CMakeLists.txt
|-- README.md
|-- resource
|   |-- dicts
|   |   `-- sch_dict.txt    // 字典
|   `-- models
|       |-- classify.onnx    // 方向分类模型
|       |-- detect.onnx    // 文字检测模型
|       `-- recognize.onnx    // 文字识别模型
`-- source
    |-- Communicate    // socket通讯
    |   |-- CMakeLists.txt
    |   |-- server.cpp
    |   `-- server.h
    |-- Loader    // 加载模型和字典的类
    |   |-- CMakeLists.txt
    |   |-- loader.cpp
    |   `-- loader.h
    |-- ModelOP    // 模型的预处理和后处理类
    |   |-- CMakeLists.txt
    |   |-- clsnetop.cpp
    |   |-- clsnetop.h
    |   |-- dbnetop.cpp
    |   |-- dbnetop.h
    |   |-- recnetop.cpp
    |   `-- recnetop.h
    `-- main.cpp
