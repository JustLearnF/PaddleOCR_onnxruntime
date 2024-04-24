#pragma once
#include <onnxruntime_cxx_api.h>

#include <opencv2/opencv.hpp>
using namespace Ort;

class DBNetOP {
 public:
  struct Params {
    float threshold = 0.3;
    int box_threshold = 144;
    float scale_w = 1.5;
    float scale_h = 2.0;
    size_t max_candidates = 1000;
  };

 public:
  std::vector<std::unique_ptr<cv::Mat>> text_imgs;

 private:
  AllocatorWithDefaultOptions allocator;
  Session *model;
  cv::Mat input_img;
  cv::Mat *resized_img;
  cv::Mat *operating_img;
  std::vector<const char *> input_node_names;
  std::vector<const char *> output_node_names;
  ShapeInferContext::Ints input_shape;
  Params parmas;

 public:
  DBNetOP(Session *model);
  ~DBNetOP();
  bool setInputImage(std::string img_path);
  void predict();

 private:
  void preProcess();
  void postProcess(float *data);
};