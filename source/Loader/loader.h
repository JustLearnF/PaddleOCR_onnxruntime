#pragma once
#include <onnxruntime_cxx_api.h>
#include"ModelOP/clsnetop.h"
#include"ModelOP/dbnetop.h"
#include"ModelOP/recnetop.h"

#include <filesystem>
namespace fs = std::filesystem;
using namespace Ort;

#define LOADER_MODELS_PATH "resource\\models"
#define LOADER_DICTS_PATH "resource\\dicts"

class Loader {
 public:
  struct Loader_Sessions {
    Session *detect = nullptr;
    Session *classify = nullptr;
    Session *recognize = nullptr;
  };

  struct Loader_Names {
    std::string det_name = "detect.onnx";
    std::string cls_name = "classify.onnx";
    std::string rec_name = "recognize.onnx";
    std::string dict_name = "sch_dict.txt";
  };

  struct Loader_Results
  {
    cv::Mat *resized_img;
    std::vector<cv::Rect> *rects;
    std::vector<std::unique_ptr<cv::Mat>> *text_imgs;
    std::vector<std::unique_ptr<std::string>> *texts;
  };

 private:
  fs::path current_path;
  Env *env;
  Loader_Sessions sessions;
  std::string models_path = LOADER_MODELS_PATH;
  std::string dicts_path = LOADER_DICTS_PATH;
  Loader_Names loader_names;
  std::vector<std::string> dict;
  DBNetOP *dbnetop;
  ClsNetOP *clsnetop;
  RecNetOP *recnetop;

 public:
  Loader();
  ~Loader();
  bool load();
  void release();
  Loader_Names *getLoaderNames();  // 预留接口以便修改模型名和字典名
  const Loader_Results predict(std::string img_path);

 private:
  bool isModelsExist();
  bool loadModel(std::string model_name, Session *&model);
  bool isDictExist();
  bool loadDict();
  bool loadModelOP();
};