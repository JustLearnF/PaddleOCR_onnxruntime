#pragma once
#include <onnxruntime_cxx_api.h>

#include <opencv2/opencv.hpp>
using namespace Ort;

class RecNetOP {
 public:
  struct Params {
    int minimum_sizeH = 48;
  };

 public:
  std::vector<std::unique_ptr<std::string>> texts;

 private:
  AllocatorWithDefaultOptions allocator;
  Session *model;
  std::vector<std::unique_ptr<cv::Mat>> *text_imgs;
  std::vector<const char *> input_node_names;
  std::vector<const char *> output_node_names;
  ShapeInferContext::Ints input_shape;
  Params params;
  std::vector<std::string> *dict;

 public:
  RecNetOP(Session *model, std::vector<std::string> *dict);
  ~RecNetOP();
  void setTextImages(std::vector<std::unique_ptr<cv::Mat>> *text_imgs);
  void predict();

 private:
  void preProcess();
  void postProcess(float *data, ShapeInferContext::Ints shape);
};