#include "dbnetop.h"

#include <cmath>
#include <opencv2/core/utils/logger.hpp>

#define MINIMUM_SIZE 32

DBNetOP::DBNetOP(Session *model) : model(model) {
  cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
  std::string *input_name = new std::string(
      this->model->GetInputNameAllocated(0, this->allocator).get());
  std::string *output_name = new std::string(
      this->model->GetOutputNameAllocated(0, this->allocator).get());
  this->input_node_names.push_back(input_name->c_str());
  this->output_node_names.push_back(output_name->c_str());
  this->input_shape =
      this->model->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  this->input_shape[0] = 1;
}

DBNetOP::~DBNetOP() {}

bool DBNetOP::setInputImage(std::string img_path) {
  if (!this->input_img.empty()) this->input_img.release();
  if (!this->resized_img && !this->resized_img->empty()) {
    this->resized_img->release();
    this->resized_img = nullptr;
  }
  this->input_img = cv::imread(img_path);
  if (this->input_img.empty()) return false;
  this->text_imgs.clear();
  this->rects.clear();
  return true;
}

void DBNetOP::predict() {
  if (this->input_img.empty()) return;
  preProcess();
  Value input_tensor = Value::CreateTensor<float>(
      this->allocator.GetInfo(),
      reinterpret_cast<float *>(this->operating_img->data),
      this->operating_img->rows * this->operating_img->cols,
      this->input_shape.data(), 4);
  auto output_tensor =
      this->model->Run(RunOptions{nullptr}, this->input_node_names.data(),
                       &input_tensor, 1, this->output_node_names.data(), 1);
  postProcess(output_tensor[0].GetTensorMutableData<float>());
}

void DBNetOP::preProcess() {
  // 计算缩放后的宽和高
  int ratio_h =
      static_cast<int>(std::round((float)this->input_img.rows / MINIMUM_SIZE));
  int ratio_w =
      static_cast<int>(std::round((float)this->input_img.cols / MINIMUM_SIZE));
  ratio_h = (ratio_h == 0) ? 1 : ratio_h;
  ratio_w = (ratio_w == 0) ? 1 : ratio_w;
  this->input_shape[2] = ratio_h * MINIMUM_SIZE;
  this->input_shape[3] = ratio_w * MINIMUM_SIZE;
  // 将bgr图像转换为rgb图像同时进行归一化
  this->resized_img = new cv::Mat();
  cv::resize(this->input_img, *this->resized_img,
             cv::Size(this->input_shape[3], this->input_shape[2]));
  cv::Mat rgb_img;
  cv::cvtColor(*this->resized_img, rgb_img, cv::COLOR_BGR2RGB);
  cv::Mat rgb_img_float;
  this->tmp_img = new cv::Mat();
  rgb_img.convertTo(*this->tmp_img, CV_32FC3);
  cv::Mat normlized_resized_img = (*this->tmp_img) / 255.0f;
  std::vector<cv::Mat> channels;
  cv::split(normlized_resized_img, channels);
  this->operating_img = new cv::Mat();
  cv::vconcat(channels, *this->operating_img);
  normlized_resized_img.release();
}

void DBNetOP::postProcess(float *data) {
  // 通过概率图获取二值图
  cv::Mat ap_binary_img(this->input_shape[2], this->input_shape[3], CV_32F,
                        data);
  cv::Mat binary_img_float;
  cv::threshold(ap_binary_img, binary_img_float, this->parmas.threshold, 255,
                cv::THRESH_BINARY);
  cv::Mat binary_img;
  binary_img_float.convertTo(binary_img, CV_8U);
  // 查找连通域以及缩放和归一化处理
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary_img, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);
  cv::Rect resized_img_rect(0, 0, this->input_shape[3], this->input_shape[2]);
  for (auto contour : contours) {
    cv::Rect bounding_rect = cv::boundingRect(contour);
    if (bounding_rect.area() >= this->parmas.box_threshold) {
      // 对目标区域进行放大 尽可能包含区域中的文字
      int offset_w = static_cast<int>(((this->parmas.scale_w - 1) / 2.0f) *
                                      bounding_rect.width);
      int offset_h = static_cast<int>(((this->parmas.scale_h - 1) / 2.0f) *
                                      bounding_rect.height);
      int newX = bounding_rect.x - offset_w;
      int newY = bounding_rect.y - offset_h;
      newX = (newX < 0) ? 0 : newX;
      newY = (newY < 0) ? 0 : newY;
      int newW = bounding_rect.width + 2 * offset_w;
      int newH = bounding_rect.height + 2 * offset_h;
      cv::Rect rect = cv::Rect(newX, newY, newW, newH) & resized_img_rect;
      cv::Mat *normlized_resized_img =
          new cv::Mat((*this->tmp_img)(rect) / 255.0f);
      this->text_imgs.push_back(
          std::unique_ptr<cv::Mat>(normlized_resized_img));
      this->rects.push_back(rect);
    }
  }
  this->operating_img->release();
  this->operating_img = nullptr;
  this->tmp_img->release();
  this->tmp_img = nullptr;
}