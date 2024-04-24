#include "server.h"

Server::Server(int port) : port(port) {}

Server::~Server() {
  closesocket(clientSocket);
  closesocket(serverSocket);
  WSACleanup();
}

bool Server::init() {
  WSADATA wsaData;
  if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
    return false;
  }
  this->serverSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  sockaddr_in serverAddr;
  serverAddr.sin_family = AF_INET;
  serverAddr.sin_addr.s_addr = INADDR_ANY;
  serverAddr.sin_port = htons(this->port);
  if (bind(this->serverSocket, (sockaddr*)&serverAddr, sizeof(serverAddr)) ==
      SOCKET_ERROR) {
    closesocket(serverSocket);
    WSACleanup();
    return false;
  }
  if (listen(serverSocket, SOMAXCONN) == SOCKET_ERROR) {
    closesocket(serverSocket);
    WSACleanup();
    return false;
  }
}

bool Server::waitForClient() {
  this->clientSocket = accept(this->serverSocket, NULL, NULL);
  if (clientSocket == INVALID_SOCKET) {
    closesocket(serverSocket);
    WSACleanup();
    return false;
  } else
    return true;
}

std::string Server::recvImagePath() {
  if (this->clientSocket == INVALID_SOCKET) return NULL;
  char recvBuffer[MAX_PATH];
  int bufferLength =
      recv(this->clientSocket, recvBuffer, sizeof(recvBuffer), 0);
  if (bufferLength == SOCKET_ERROR) return NULL;
  return std::string(recvBuffer,bufferLength);
}
