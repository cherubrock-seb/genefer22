#include <memory>
#include <iostream>
#include <cstdlib>
#include "ocl.h"
#include "transform.h"
int main(int argc, char**argv) {
  if (argc!=2) return 2;
  auto plat=reinterpret_cast<cl_platform_id>(1);
  auto dev=reinterpret_cast<cl_device_id>(2);
  try {
    std::unique_ptr<transform> tr(transform::create_gpu(1024, static_cast<unsigned>(std::stoi(argv[1])), true, 0, 2, plat, dev, false));
    tr->squareDup(true); tr->initMultiplicand(1); tr->mul(); std::cout<<"DONE\n";
  } catch(const std::exception& e) {std::cout<<"ERROR "<<e.what()<<"\n";return 3;}
}
