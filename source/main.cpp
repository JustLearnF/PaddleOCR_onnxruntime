// #include "Loader/loader.h"
// #include "ModelOP/clsnetop.h"
// #include "ModelOP/dbnetop.h"
// #include "ModelOP/recnetop.h"
#include <Communicate/server.h>

#include <iostream>

int main() {
  // Loader *loader = new Loader();
  // loader->load();
  // DBNetOP *dbnetop = new DBNetOP(loader->getModel(1));
  // dbnetop->setInputImage("./test/image_test.jpg");
  // dbnetop->predict();
  // ClsNetOP *clsnetop = new ClsNetOP(loader->getModel(2));
  // clsnetop->setTextImages(&dbnetop->text_imgs);
  // clsnetop->predict();
  // RecNetOP *recnetop = new RecNetOP(loader->getModel(3), loader->getDict());
  // recnetop->setTextImages(&dbnetop->text_imgs);
  // recnetop->predict();
  Server *server = new Server();
  server->init();
  server->waitForClient();
  std::cout << server->recvImagePath() << std::endl;
  return 0;
}