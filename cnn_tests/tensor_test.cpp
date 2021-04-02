#include <iostream>
#include <Eigen>
#include <Eigen/CXX11/Tensor>
using namespace std;
using namespace Eigen;

int main(int argc, char *argv[]){
  // Eigen::array<IndexPair<int>, 1> dims = { IndexPair<int>(2, 0), IndexPair<int>(1, 1)};
  // Tensor<float, 3> t1(2, 3, 4);
  // Tensor<float, 3> t2(4, 3, 3);
  //
  // Eigen::Tensor<float, 2> o = t1.contract(t2, dims);

  Eigen::Tensor<int, 2> a(4, 3);
  a.setValues({{0, 100, 200}, {300, 400, 500}, {600, 700, 800}, {900, 1000, 1100}});
  Eigen::array<Eigen::DenseIndex, 2> strides({3, 2});
  Eigen::Tensor<int, 2> b = a.stride(strides);
  a(0, 0) = 1.0f;
  cout << "b" << endl << b << endl;
  cout << "a" << endl << a << endl;

  return 0;
}
