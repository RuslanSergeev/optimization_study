#include <iostream>
#include <array>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
using namespace std;

enum class Storage_order {
  row_major,        // C/C++/Objective-C, numpy, pyTorch, PL/I, Pascal
  col_major         // Eigen, Fortran, MATLAB, GNU Octave, R, Julia, and Scilab
};

enum class Padding_type {
    zero_pad,
    same_pad
};


class Tensor_block{
public:
    int batch_min=0, num_batch;
    int chan_min=0,  num_chan;
    int row_min=0,   num_rows;
    int col_min=0,   num_cols;

    int size() const
    {
        return num_batch*num_chan*num_rows*num_cols;
    }
};

class Tensor{
public:
    Tensor(int num_batch=0, int num_chan=0, int num_rows=0, int num_cols=0,
           Storage_order stor_order=Storage_order::row_major, float *existing_data=nullptr)
    {
        order = stor_order;
        resize(num_batch, num_chan, num_rows, num_cols, existing_data);
    }

    virtual ~Tensor()
    {
        if(own)
        {
            delete[] data;
        }
    }

    void resize(int num_batch, int num_chan, int num_rows, int num_cols,
                float *new_data=nullptr)
    {
        int new_size = num_cols*num_rows*num_chan*num_batch;

        //delete only if it is an buffer owner (not a reference)
        if(new_size!=size() and own){
            delete[] data;
        }

        data = new_data ? new_data : new float[new_size];
        own  = new_data ? false    : true;

        reshape(num_batch, num_chan, num_rows, num_cols);
    }

    //change tensor dimensions
    void transpose(const int dim1, const int dim2)
    {
        std::swap(idx[dim1], idx[dim2]);
    }

    //change the tensor form.
    void reshape(int batch, int chan, int rows, int cols)
    {
        // https://en.wikipedia.org/wiki/Row-_and_column-major_order
        stride[idx[3]] = Storage_order::col_major == order? rows : 1;
        stride[idx[2]] = Storage_order::row_major == order? cols : 1;
        stride[idx[1]] = rows*cols;
        stride[idx[0]] = rows*cols*chan;
        lens[idx[0]] = batch;
        lens[idx[1]] = chan;
        lens[idx[2]] = rows;
        lens[idx[3]] = cols;
    }

    //Gets the tensor element located at [batch, chan, row, col]
    //If index points outside the tensor, will do one of
    // - zero padding
    // - same element padding
    float &operator()(const int batch, const int chan, const int row, const int col, Padding_type pad_type=Padding_type::zero_pad)
    {
        switch(pad_type)
        {
        case Padding_type::zero_pad:
        {
            return get_zero_pad(batch, chan, row, col);
        } break;
        case Padding_type::same_pad:
        {
            return get_same_pad(batch, chan, row, col);;
        }break;
        default:{
            return get_zero_pad(batch, chan, row, col);;
        }break;
        }
    }

    //output the tensor to stdout
    void print()
    {
        for(int batch=0; batch < lens[idx[0]]; ++batch)
        {
            std::cout << "batch " << batch << std::endl;
            for(int chan=0; chan < lens[idx[1]]; ++chan)
            {
                std::cout << "  chan " << chan << std::endl;
                for(int row=0; row < lens[idx[2]]; ++row)
                {
                    std::cout << "    ";
                    for(int col=0; col < lens[idx[3]]; ++col)
                    {
                        std::cout << this->operator()(batch, chan, row, col) << " ";
                    }
                    std::cout << std::endl;
                }
            }
        }
    }

    int size(){
        return std::accumulate(lens.begin(), lens.end(), 1, [](int a, int b){
            return a * b;
        });
    }

    std::array<int, 4> stride;
    std::array<int, 4> lens = {0, 0, 0, 0};
    std::array<int, 4> idx  = {0, 1, 2, 3};
    Storage_order order;
    float *data = nullptr;
    bool own=false;

private:
    float zero=0;

    //get an element in case of zero-padding
    float &get_zero_pad(const int batch, const int chan, const int row, const int col)
    {
        if(batch < 0 or batch>=lens[idx[0]] or chan  < 0 or chan >=lens[idx[1]] or
           row   < 0 or row  >=lens[idx[2]]  or col  < 0 or col  >=lens[idx[3]])
        {
            zero=0;
            return zero;
        }
        return get_no_pad(batch, chan, row, col);
    }

    //get an element in case of same element padding
    float &get_same_pad(const int batch, const int chan, const int row, const int col)
    {
        auto new_batch = std::min(lens[idx[0]]-1, std::max(0, batch));
        auto new_chan  = std::min(lens[idx[1]]-1, std::max(0, chan));
        auto new_row   = std::min(lens[idx[2]]-1, std::max(0, row));
        auto new_col   = std::min(lens[idx[3]]-1, std::max(0, col));

        return get_no_pad(new_batch, new_chan, new_row, new_col);
    }

    //get an element considering no padding (padding done before).
    //Works with row and col-major tensors.
    float &get_no_pad(const int batch, const int chan, const int row, const int col)
    {
        int offset = col*stride[idx[3]] + row*stride[idx[2]] + chan*stride[idx[1]] + batch*stride[idx[0]];
        return data[offset];
    }
};


//Calculation of the output tensor sizes,
//with given convolution/pooling parameters.
Tensor_block calc_size(const int h_in, const int w_in,
                       const int k_h, const int k_w,
                       const int str_h, const int str_w,
                       const int pad_h, const int pad_w,
                       const int dil_h, const int dil_w)
{
    Tensor_block o;
    //result tensor num rows
    o.num_rows = int((h_in+2*pad_h-dil_h*(k_h-1)-1)/str_h+1);
    //result tensor num cols
    o.num_cols = int((w_in+2*pad_w-dil_w*(k_w-1)-1)/str_w+1);
    return o;
}

void cnn2d(const int str_h, const int str_w,
           const int pad_h, const int pad_w, Padding_type ptype,
           const int dil_h, const int dil_w,
           Tensor &bias,
           Tensor &kern,
           Tensor &src,
           Tensor &dst)
{
    int out_ch = kern.lens[0];
    int in_ch = kern.lens[1];
    int k_h = kern.lens[2];
    int k_w = kern.lens[3];

    auto size = calc_size(src.lens[2], src.lens[3], k_h, k_w,
                          str_h, str_w, pad_h, pad_w, dil_h, dil_w);
    dst.resize(src.lens[0], out_ch, size.num_rows, size.num_cols);

    //initialize output
    for(int cur_out=0; cur_out < out_ch; ++cur_out){
        //initialize with bias value.
        for(int cur_h=0; cur_h < size.num_rows; ++cur_h){
            for(int cur_w=0; cur_w < size.num_cols; ++cur_w){
                dst(0, cur_out, cur_h, cur_w) = bias(0, 0, 0, cur_out);
            }
        }
    }

    //convolution
    for(int cur_out=0; cur_out < out_ch; ++cur_out){
        for(int cur_h=0; cur_h < size.num_rows; ++cur_h){
            for(int cur_w=0; cur_w < size.num_cols; ++cur_w){
                //actual cross-convolution kernel.
                for(int cur_in=0; cur_in < in_ch; ++cur_in){
                    for(int cur_kh=0; cur_kh < k_h; ++cur_kh){
                        for(int cur_kw=0; cur_kw < k_w; ++cur_kw){
                            dst(0, cur_out, cur_h, cur_w) +=
                                    src(0, cur_in,
                                        cur_h*str_h+cur_kh*dil_h-pad_h,
                                        cur_w*str_w+cur_kw*dil_w-pad_w,
                                        ptype) *
                                    kern(cur_out, cur_in, cur_kh, cur_kw);
                        }//kw
                    }//kh
                }//cur_in
            }//cur_w
        }//cur_h
    }//cur_out
}


void maxpool2d(const int k_h, const int k_w,
               const int str_h, const int str_w,
               const int pad_h, const int pad_w,
               const int dil_h, const int dil_w,
               Tensor &src,
               Tensor &dst)
{
    int out_ch = src.lens[1];

    auto size = calc_size(src.lens[2], src.lens[3], k_h, k_w,
                          str_h, str_w, pad_h, pad_w, dil_h, dil_w);
    dst.resize(src.lens[0], out_ch, size.num_rows, size.num_cols);

    //initialize output
    for(int cur_out=0; cur_out < out_ch; ++cur_out){
        //initialize with bias value.
        for(int cur_h=0; cur_h < size.num_rows; ++cur_h){
            for(int cur_w=0; cur_w < size.num_cols; ++cur_w){
                dst(0, cur_out, cur_h, cur_w) = std::numeric_limits<float>::lowest();
            }
        }
    }

    //convolution
    for(int cur_out=0; cur_out < out_ch; ++cur_out){
        for(int cur_h=0; cur_h < size.num_rows; ++cur_h){
            for(int cur_w=0; cur_w < size.num_cols; ++cur_w){
                //actual cross-convolution kernel.
                for(int cur_kh=0; cur_kh < k_h; ++cur_kh){
                    for(int cur_kw=0; cur_kw < k_w; ++cur_kw){
                        dst(0, cur_out, cur_h, cur_w) =
                                std::max(dst(0, cur_out, cur_h, cur_w),
                                         src(0, cur_out,
                                             cur_h*str_h+cur_kh*dil_h-pad_h,
                                             cur_w*str_w+cur_kw*dil_w-pad_w));
                    }//kw
                }//kh
            }//cur_w
        }//cur_h
    }//cur_out
}

void avgpool2d(const int k_h, const int k_w,
               const int str_h, const int str_w,
               const int pad_h, const int pad_w,
               const int dil_h, const int dil_w,
               Tensor &src,
               Tensor &dst)
{
    int out_ch = src.lens[1];
    int kernel_size = k_h*k_w;
    auto size = calc_size(src.lens[2], src.lens[3], k_h, k_w,
                          str_h, str_w, pad_h, pad_w, dil_h, dil_w);
    dst.resize(src.lens[0], out_ch, size.num_rows, size.num_cols);

    //initialize output
    for(int cur_out=0; cur_out < out_ch; ++cur_out){
        //initialize with bias value.
        for(int cur_h=0; cur_h < size.num_rows; ++cur_h){
            for(int cur_w=0; cur_w < size.num_cols; ++cur_w){
                dst(0, cur_out, cur_h, cur_w) = 0;
            }
        }
    }

    //convolution
    for(int cur_out=0; cur_out < out_ch; ++cur_out){
        for(int cur_h=0; cur_h < size.num_rows; ++cur_h){
            for(int cur_w=0; cur_w < size.num_cols; ++cur_w){
                //actual cross-convolution kernel.
                for(int cur_kh=0; cur_kh < k_h; ++cur_kh){
                    for(int cur_kw=0; cur_kw < k_w; ++cur_kw){
                        dst(0, cur_out, cur_h, cur_w) +=
                                src(0, cur_out,
                                    cur_h*str_h+cur_kh*dil_h-pad_h,
                                    cur_w*str_w+cur_kw*dil_w-pad_w);
                    }//kw
                }//kh
                dst(0, cur_out, cur_h, cur_w) /= kernel_size;
            }//cur_w
        }//cur_h
    }//cur_out
}


#ifdef TENSOR_CNN_TESTING
int main(int argc, char *argv[])
{

  float d[48];
  for(int i = 0; i < 48; ++i){
    d[i] = i+1;
  }

  Tensor t(2, 2, 3, 4, Storage_order::row_major, d);
  t.reshape(2, 2, 1, 12);
  t.reshape(1, 1, 1, 48);
  // t.reshape(1, 1, 4, 12);
  // Tensor tt(1, 1, 4, 12, Storage_order::row_major, d);
  t.print();

  // const int in_ch = 1;
  // const int out_ch = 3;
  // const int k_h=3, k_w=3;
  // const int strd_h=2, strd_w=1;
  // const int pad_h=2, pad_w=1;
  // const int dil_h=2, dil_w=1;
  // const int in_h = 6;
  // const int in_w = 6;
  //
  // Tensor src(1, in_ch, in_h, in_w);
  // for(int ch = 0; ch < in_ch; ++ch){
  //   float el = 0;
  //   for(int cur_h = 0; cur_h < in_h; ++cur_h){
  //     for(int cur_w = 0; cur_w < in_w; ++cur_w){
  //       src(0, ch, cur_h, cur_w) = ++el;
  //     }
  //   }
  // }
  //
  // Tensor kernels(out_ch, in_ch, k_h, k_w);
  // float kvalue = 0;
  // for(int cur_out=0; cur_out<out_ch; ++cur_out){
  //   for(int cur_in=0; cur_in<in_ch; ++cur_in){
  //     for(int cur_h = 0; cur_h < k_h; ++cur_h){
  //       for(int cur_w=0; cur_w < k_w; ++cur_w){
  //           kernels(cur_out, cur_in, cur_h, cur_w) = kvalue++;
  //       }
  //     }
  //   }
  // }
  //
  // Tensor bias(1, 1, 1, out_ch);
  // for(int cur_out=0; cur_out<out_ch; ++cur_out){
  //   bias(0, 0, 0, cur_out) = cur_out;
  // }
  //
  // Tensor dst;
  //
  // cnn2d(strd_h, strd_w,
  //       pad_h, pad_w, Padding_type::zero_pad,
  //       dil_h, dil_w,
  //       bias, kernels, src, dst);
  //
  // kernels.print();
  // bias.print();
  // src.print();
  // dst.print();
  //
  // Tensor dst_pool;
  // maxpool2d(2, 2,
  //           2, 2,
  //           0, 0,
  //           1, 1,
  //           dst, dst_pool);
  // dst_pool.print();
  //
  // Tensor avg_pool;
  // avgpool2d(2, 2,
  //           2, 2,
  //           0, 0,
  //           1, 1,
  //           dst, avg_pool);
  // avg_pool.print();

  return 0;
}
#endif
