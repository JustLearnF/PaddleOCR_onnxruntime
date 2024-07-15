#include "loader.h"

#include <Windows.h>

#include <fstream>

Loader::Loader() {
  this->env = new Env(ORT_LOGGING_LEVEL_WARNING, "OCR");
#ifndef _DEBUG
  char buffer[MAX_PATH];
  GetModuleFileNameA(NULL, buffer, MAX_PATH);
  std::string fullPath(buffer);
  this->current_path =
      fs::path(fullPath.substr(0, fullPath.find_last_of("\\/")));
#else
  this->current_path = fs::current_path();
#endif
}

Loader::~Loader() { release(); }

bool Loader::load() {
  if (!isModelsExist()) return false;
  release();
  return (loadModel(this->loader_names.det_name, this->sessions.detect) &&
          loadModel(this->loader_names.cls_name, this->sessions.classify) &&
          loadModel(this->loader_names.rec_name, this->sessions.recognize) &&
          loadDict() && loadModelOP());
}

void Loader::release() {
  if (this->sessions.detect != nullptr) this->sessions.detect->release();
  if (this->sessions.classify != nullptr) this->sessions.classify->release();
  if (this->sessions.recognize != nullptr) this->sessions.recognize->release();
}

Loader::Loader_Names* Loader::getLoaderNames() { return &this->loader_names; }

const Loader::Loader_Results Loader::predict(std::string img_path) {
  if (!this->dbnetop->setInputImage(img_path)) return Loader::Loader_Results();
  this->dbnetop->predict();
  this->clsnetop->setTextImages(&this->dbnetop->text_imgs);
  this->clsnetop->predict();
  this->recnetop->setTextImages(&this->dbnetop->text_imgs);
  this->recnetop->predict();
  if (!this->recnetop->texts.empty()) {
    Loader::Loader_Results results;
    results.resized_img = this->dbnetop->resized_img;
    results.rects = &this->dbnetop->rects;
    results.text_imgs = &this->dbnetop->text_imgs;
    results.texts = &this->recnetop->texts;
    return results;
  } else
    return Loader::Loader_Results();
}

bool Loader::isModelsExist() {
  return (fs::exists(this->current_path / this->models_path /
                     this->loader_names.det_name) &&
          fs::exists(this->current_path / this->models_path /
                     this->loader_names.cls_name) &&
          fs::exists(this->current_path / this->models_path /
                     this->loader_names.rec_name));
}

bool Loader::loadModel(std::string model_name, Session*& model) {
  SessionOptions options;
  model = new Session(
      *this->env, (this->current_path / this->models_path / model_name).c_str(),
      options);
  if (model != nullptr) return true;
  return false;
}

bool Loader::isDictExist() {
  return fs::exists(this->current_path / this->dicts_path /
                    this->loader_names.dict_name);
}

bool Loader::loadDict() {
  if (!isDictExist()) return false;
  std::ifstream ifs(this->current_path / this->dicts_path /
                    this->loader_names.dict_name);
  std::string line;
  while (std::getline(ifs, line)) {
    this->dict.push_back(line);
  }
  this->dict.push_back(" ");
  return true;
}

bool Loader::loadModelOP() {
  if (!this->sessions.classify || !this->sessions.detect ||
      !this->sessions.recognize || this->dict.empty())
    return false;
  this->dbnetop = new DBNetOP(this->sessions.detect);
  this->clsnetop = new ClsNetOP(this->sessions.classify);
  this->recnetop = new RecNetOP(this->sessions.recognize, &this->dict);
  return true;
}
