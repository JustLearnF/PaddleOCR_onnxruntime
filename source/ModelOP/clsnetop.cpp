#include "clsnetop.h"

ClsNetOP::ClsNetOP(Session *model) : model(model) {
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

ClsNetOP::~ClsNetOP() {}

void ClsNetOP::setTextImages(std::vector<std::unique_ptr<cv::Mat>> *text_imgs) {
  this->text_imgs = text_imgs;
}

void ClsNetOP::predict() {
  if (this->text_imgs->empty()) return;
  for (auto &text_img_ptr : *this->text_imgs) {
    cv::Mat *text_img = text_img_ptr.get();
    // 对文本图像进行缩放以获取到所需大小的图像
    cv::Mat *resized_img = preProcess(
        *text_img);  // 返回的是新图像 因为cls模型和rec模型所需大小不一
    this->input_shape[2] = resized_img->rows / 3;
    this->input_shape[3] = resized_img->cols;
    try {
      Value input_tensor = Value::CreateTensor<float>(
          this->allocator.GetInfo(),
          reinterpret_cast<float *>(resized_img->data),
          resized_img->rows * resized_img->cols, this->input_shape.data(), 4);
      auto output_tensor =
          this->model->Run(RunOptions{nullptr}, this->input_node_names.data(),
                           &input_tensor, 1, this->output_node_names.data(), 1);
      float *result = output_tensor[0].GetTensorMutableData<float>();
      if (result[1] > result[0]) {
        cv::rotate(*text_img, *text_img,
                   cv::ROTATE_180);  // 如果图像是180°的概率大于0°就翻转
      }
      input_tensor.release();
      output_tensor[0].release();
      output_tensor.clear();
      resized_img->release();
    } catch (const Exception &e) {
      std::cerr << e.what() << '\n';
    }
  }
}

cv::Mat *ClsNetOP::preProcess(cv::Mat &img) {
  cv::Mat resized_img_;
  cv::resize(img, resized_img_,
             cv::Size(this->params.maximum_w, this->params.maximum_h),
             cv::INTER_LINEAR);
  std::vector<cv::Mat> channels;
  cv::split(resized_img_, channels);
  cv::Mat *resized_img = new cv::Mat();
  cv::vconcat(channels, *resized_img);
  return resized_img;
}