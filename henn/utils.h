#pragma once

#include <iostream>
#include <sys/resource.h>
void printMemoryUsage() {
  struct rusage r;
  getrusage(RUSAGE_SELF, &r);
  std::cout << "Max resident memory: " << r.ru_maxrss/1024 << " MB" << std::endl;
}
