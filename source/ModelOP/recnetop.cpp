#include "recnetop.h"

#include <algorithm>

RecNetOP::RecNetOP(Session* model, std::vector<std::string>* dict)
    : model(model), dict(dict) {
  std::string* input_name = new std::string(
      this->model->GetInputNameAllocated(0, this->allocator).get());
  std::string* output_name = new std::string(
      this->model->GetOutputNameAllocated(0, this->allocator).get());
  this->input_node_names.push_back(input_name->c_str());
  this->output_node_names.push_back(output_name->c_str());
  this->input_shape =
      this->model->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  this->input_shape[0] = 1;
  this->input_shape[2] = this->params.minimum_sizeH;
}

RecNetOP::~RecNetOP() {}

void RecNetOP::setTextImages(std::vector<std::unique_ptr<cv::Mat>>* text_imgs) {
  this->text_imgs = text_imgs;
}

void RecNetOP::predict() {
  if (this->text_imgs->empty()) return;
  this->results.clear();
  // 对文本图像进行缩放和归一化
  preProcess();
  for (auto& text_img_ptr : *this->text_imgs) {
    cv::Mat text_img = *text_img_ptr.get();
    this->input_shape[3] = text_img.cols;
    Value input_tensor = Value::CreateTensor<float>(
        this->allocator.GetInfo(), reinterpret_cast<float*>(text_img.data),
        text_img.rows * text_img.cols, this->input_shape.data(), 4);
    auto output_tensor =
        this->model->Run(RunOptions{nullptr}, this->input_node_names.data(),
                         &input_tensor, 1, this->output_node_names.data(), 1);
    auto output_shape = output_tensor[0].GetTensorTypeAndShapeInfo().GetShape();
    // 处理结果
    postProcess(output_tensor[0].GetTensorMutableData<float>(), output_shape);
    input_tensor.release();
    output_tensor[0].release();
  }
}

void RecNetOP::preProcess() {
  for (auto& text_img_ptr : *this->text_imgs) {
    cv::Mat* text_img = text_img_ptr.get();
    cv::Mat resize_img;
    cv::resize(*text_img, resize_img,
               cv::Size(text_img->cols, this->params.minimum_sizeH),
               cv::INTER_LINEAR);
    std::vector<cv::Mat> channels;
    cv::split(resize_img, channels);
    cv::Mat* chw_img = new cv::Mat();
    cv::vconcat(channels, *chw_img);
    text_img->release();
    resize_img.release();
    text_img_ptr = std::unique_ptr<cv::Mat>(chw_img);
  }
}

void RecNetOP::postProcess(float* data, ShapeInferContext::Ints shape) {
  std::string* text = new std::string();
  // 获取结果中概率最大的下标 然后查询字典将文字存储
  for (int i = 0; i < shape[1]; i++) {
    int label =
        std::max_element(&data[i * shape[2]], &data[(i + 1) * shape[2]]) -
        &data[i * shape[2]];
    if (label != 0) *text += (*this->dict)[label - 1];
  }
  this->results.push_back(std::unique_ptr<std::string>(text));
}
