#pragma once
#include <WinSock2.h>

#include <string>
#pragma comment(lib, "ws2_32.lib")

class Server {
 private:
  int port;
  SOCKET serverSocket;
  SOCKET clientSocket;

 public:
  Server(int port = 48623);
  ~Server();
  bool init();
  bool waitForClient();
  std::string recvImagePath();
};
