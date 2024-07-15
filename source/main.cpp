#include "Loader/loader.h"

#include <iostream>

int main() {
  Loader *loader = new Loader();
  loader->load();
  auto results = loader->predict("./test/image_test.jpg");
  return 0;
}