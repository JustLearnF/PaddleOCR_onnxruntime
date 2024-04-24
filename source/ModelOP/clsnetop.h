#pragma once
#include <onnxruntime_cxx_api.h>

#include <opencv2/opencv.hpp>
using namespace Ort;

class ClsNetOP {
 public:
  struct Params {
    int maximum_w = 192;
    int maximum_h = 48;
  };

 private:
  AllocatorWithDefaultOptions allocator;
  Session *model;
  std::vector<std::unique_ptr<cv::Mat>> *text_imgs;
  std::vector<const char *> input_node_names;
  std::vector<const char *> output_node_names;
  ShapeInferContext::Ints input_shape;
  Params params;

 public:
  ClsNetOP(Session *model);
  ~ClsNetOP();
  void setTextImages(std::vector<std::unique_ptr<cv::Mat>> *text_imgs);
  void predict();

 private:
  cv::Mat *preProcess(cv::Mat &img);
};