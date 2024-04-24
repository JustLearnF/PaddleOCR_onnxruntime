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
          loadDict());
}

void Loader::release() {
  if (this->sessions.detect != nullptr) this->sessions.detect->release();
  if (this->sessions.classify != nullptr) this->sessions.classify->release();
  if (this->sessions.recognize != nullptr) this->sessions.recognize->release();
}

Loader::Loader_Names* Loader::getLoaderNames() { return &this->loader_names; }

Session* Loader::getModel(int id) {
  switch (id) {
    case 1:
      return this->sessions.detect;
      break;

    case 2:
      return this->sessions.classify;
      break;

    case 3:
      return this->sessions.recognize;
      break;

    default:
      return nullptr;
      break;
  }
}

std::vector<std::string>* Loader::getDict() { return &this->dict; }

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
