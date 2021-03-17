#include <iostream>
#include <Eigen>
using namespace Eigen;
using namespace std;

enum class padding_type {
  zero_pad,
  same_pad
};


class Conv2d {
public:
  Conv2d(const int _in_ch, const int _out_ch,
         const int _k_h, const int _k_w,
         const int _pad_h=0, const int _pad_w=0,
         const padding_type _ptype = padding_type::zero_pad,
         const int _str_h=1, const int _str_w=1,
         const int _dil_h=1, const int _dil_w=1):
     in_ch(_in_ch),
     out_ch(_out_ch),
     k_h(_k_h),
     k_w(_k_w),
     pad_h(_pad_h),
     pad_w(_pad_w),
     ptype(_ptype),
     str_h(_str_h),
     str_w(_str_w),
     dil_h(_dil_h),
     dil_w(_dil_w)
  {
  }

  MatrixXf *operator()(const MatrixXf *src,
             MatrixXf *dst=nullptr)
  {
    h_in = src->rows();
    w_in = src->cols();
    calc_size(h_in, w_in,
              k_h, k_w,
              str_h, str_w,
              pad_h, pad_w,
              dil_h, dil_w);

    MatrixXf *new_src = const_cast<MatrixXf*>(src);
    //append padding
    if(pad_h!=0 or pad_w!=0){
      new_src = new MatrixXf[in_ch];
      for(int cur_ch = 0; cur_ch < in_ch; ++cur_ch){
        new_src[cur_ch].resize(src[cur_ch].rows()+2*pad_h,
                               src[cur_ch].cols()+2*pad_w);
      }
      append_padding(in_ch, pad_h, pad_w,
                     src, new_src, ptype);
    }

    //allocate destination tensor
    if(!dst){
      dst = new MatrixXf[out_ch];
    }

    //resize output tensors for each output channel
    for(int cur_out=0; cur_out < out_ch; ++cur_out){
      //resize if needed.
      if(dst[cur_out].rows()!=h_out or dst[cur_out].cols()!=w_out){
        dst[cur_out].resize(h_out, w_out);
      }
    }

    dst = cnn2d(in_ch, out_ch,
               k_h, k_w,
               pad_h, pad_w, ptype,
               str_h, str_w,
               dil_h, dil_w,
               bias, kern,
               new_src, dst);

    if(pad_h!=0 or pad_w!=0){
     delete[] new_src;
    }
    return dst;
  }

  void set_kernels(MatrixXf *kernels)
  {
    kern = kernels;
  }

  void set_bias(VectorXf *new_bias)
  {
    bias = new_bias;
  }

  const int in_ch, out_ch;
  const int k_h, k_w;
  const int pad_h, pad_w;
  padding_type ptype;
  const int str_h, str_w;
  const int dil_h, dil_w;

  int h_out, w_out;
  int h_in, w_in;

  VectorXf *bias;
  MatrixXf *kern;

private:

  void calc_size(const int h_in, const int w_in,
                 const int k_h, const int k_w,
                 const int str_h, const int str_w,
                 const int pad_h, const int pad_w,
                 const int dil_h, const int dil_w)
  {
    h_out = int((h_in+2*pad_h-dil_h*(k_h-1)-1)/str_h+1);
    w_out = int((w_in+2*pad_w-dil_w*(k_w-1)-1)/str_w+1);
  }

  //Добавить к исходному набору данных падинги заданного размера.
  void append_padding(const int in_ch,
                      const int pad_h,
                      const int pad_w,
                      const MatrixXf *src,
                      MatrixXf *dst,
                      padding_type type)
  {
    for(int cur_ch = 0; cur_ch < in_ch; ++cur_ch){
      for(int cur_h = 0; cur_h < dst->rows(); ++cur_h){
        for(int cur_w = 0; cur_w < dst->cols(); ++cur_w){
          int in_h = std::max(cur_h-(pad_h), 0);
          in_h = std::min(in_h, int(src->rows()-1));
          int in_w = std::max(cur_w-(pad_w), 0);
          in_w = std::min(in_w, int(src->cols()-1));

          if(cur_h < pad_h or cur_h >= pad_h+src->rows() or
             cur_w < pad_w or cur_w >= pad_w+src->cols()){
             switch(type){
               case padding_type::zero_pad:{
                 dst[cur_ch](cur_h, cur_w) = 0;
                 break;
               }
               case padding_type::same_pad:{
                 dst[cur_ch](cur_h, cur_w) = src[cur_ch](in_h, in_w);
                 break;
               }
             }//switch
          }//if
          else{
            dst[cur_ch](cur_h, cur_w) = src[cur_ch](in_h, in_w);
          }
        }
      }
    }
  }

  MatrixXf *cnn2d(const int in_ch, const int out_ch,
             const int k_h, const int k_w,
             const int pad_h, const int pad_w, padding_type ptype,
             const int str_h, const int str_w,
             const int dil_h, const int dil_w,
             const VectorXf *bias,
             const MatrixXf *kern,
             const MatrixXf *src,
             MatrixXf *dst=nullptr)
  {
    //initialize output
    for(int cur_out=0; cur_out < out_ch; ++cur_out){
      //initialize with bias value.
      for(int cur_h=0; cur_h < h_out; ++cur_h){
        for(int cur_w=0; cur_w < w_out; ++cur_w){
          dst[cur_out](cur_h, cur_w) = (*bias)(cur_out);
        }
      }
    }
    //convolution
    for(int cur_out=0; cur_out < out_ch; ++cur_out){
      for(int cur_h=0; cur_h < h_out; ++cur_h){
        for(int cur_w=0; cur_w < w_out; ++cur_w){
          //actual cross-convolution kernel.
          for(int cur_in=0; cur_in < in_ch; ++cur_in){
            for(int cur_kh=0; cur_kh < k_h; ++cur_kh){
              for(int cur_kw=0; cur_kw < k_w; ++cur_kw){
                dst[cur_out](cur_h, cur_w) +=
                src[cur_in](cur_h*str_h+cur_kh*dil_h, cur_w*str_w+cur_kw*dil_w) *
                kern[cur_out*in_ch+cur_in](cur_kh, cur_kw);
              }//kw
            }//kh
          }//cur_in
        }//cur_w
      }//cur_h
    }//cur_out

    return dst;
  }
};


int main(int argc, char *argv[])
{
  const int in_ch = 1;
  const int out_ch = 3;
  const int k_h=3, k_w=3;
  const int strd_h=2, strd_w=1;
  const int pad_h=2, pad_w=1;
  const int dil_h=2, dil_w=1;
  const int in_h = 6;
  const int in_w = 6;

  MatrixXf *src = new MatrixXf[in_ch];
  for(int ch = 0; ch < in_ch; ++ch){
    src[ch].resize(in_h, in_w);
    float el = 0;
    for(int cur_h = 0; cur_h < in_h; ++cur_h){
      for(int cur_w = 0; cur_w < in_w; ++cur_w){
        src[ch](cur_h, cur_w) = ++el;
      }
    }
  }

  MatrixXf *kernels = new MatrixXf[in_ch*out_ch];
  float kvalue = 0;
  for(int cur_out=0; cur_out<out_ch; ++cur_out){
    for(int cur_in=0; cur_in<in_ch; ++cur_in){
      kernels[cur_out*in_ch + cur_in].resize(k_h, k_w);
      for(int cur_h = 0; cur_h < k_h; ++cur_h){
        for(int cur_w=0; cur_w < k_w; ++cur_w){
          kernels[cur_out*in_ch + cur_in](cur_h, cur_w) = kvalue++;
        }
      }
    }
  }

  VectorXf *bias = new VectorXf(out_ch);
  for(int cur_out=0; cur_out<out_ch; ++cur_out){
    bias[0](cur_out) = cur_out;
  }

  Conv2d cnn(in_ch, out_ch,
              k_h, k_w,
              pad_h, pad_w, padding_type::zero_pad,
              strd_h, strd_w,
              dil_h, dil_w);

  cnn.set_bias(bias);
  cnn.set_kernels(kernels);

  auto dst = cnn(src);

  for(int cur_out = 0; cur_out < out_ch; ++cur_out){
    cout << "channel: " << cur_out << endl;
    cout << dst[cur_out] << endl;
  }

  return 0;
}
