#include <iostream>
#include <array>
#include <iostream>
#include <limits>
#include <numeric>

enum class Storage_order {
  row_major,        // C/C++/Objective-C, numpy, PyTorch, PL/I, Pascal
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

    int numel() const
    {
        return num_batch*num_chan*num_rows*num_cols;
    }
};

class Tensor{
public:
    Tensor(int num_batch=0, int num_chan=0, int num_rows=0, int num_cols=0,
           Storage_order new_order=Storage_order::row_major, float *existing_data=nullptr)
    {
        resize(num_batch, num_chan, num_rows, num_cols, new_order, existing_data);
    }

    virtual ~Tensor()
    {
        if(own)
        {
            delete[] data;
        }
    }

    void resize(int num_batch, int num_chan, int num_rows, int num_cols,
                Storage_order new_order = Storage_order::row_major,
                float *new_data=nullptr)
    {
        int new_size = num_cols*num_rows*num_chan*num_batch;
        order = new_order;

        //delete only if it is an buffer owner (not a reference)
        if(new_size!=numel() and own){
            delete[] data;
        }

        data = new_data ? new_data : new float[new_size];
        own  = new_data ? false    : true;

        // https://en.wikipedia.org/wiki/Row-_and_column-major_order
        stride[3] = Storage_order::col_major == order? num_rows : 1;
        stride[2] = Storage_order::row_major == order? num_cols : 1;
        stride[1] = num_rows*num_cols;
        stride[0] = num_rows*num_cols*num_chan;

        shape[0] = num_batch;
        shape[1] = num_chan;
        shape[2] = num_rows;
        shape[3] = num_cols;
    }

    //change tensor dimensions
    Tensor &transpose(const int dim1, const int dim2)
    {
        std::swap(stride[dim1], stride[dim2]);
        std::swap(shape[dim1], shape[dim2]);

        return *this;
    }

    // reshape(*shape) -> Tensor
    //
    // Returns a tensor copy with the same data and number of elements as :attr:`self`
    // but with the specified shape.
    Tensor reshape(int batch, int chan, int rows, int cols,
                  Storage_order new_order=Storage_order::row_major)
    {
        Tensor tcopy(batch, chan, rows, cols, new_order);
        int idst = 0;
        for(int b=0; b<shape[0]; ++b)
          for(int c=0; c<shape[1]; ++c)
            for(int row=0; row<shape[2]; ++row)
              for(int col=0; col<shape[3]; ++col){
                tcopy.data[idst] = operator()(b, c, row, col);
                ++idst;
              }

        return tcopy;
    }

    // view(*shape) -> Tensor
    //
    // Returns a new tensor with the same data as the :attr:`self` tensor but of a
    // different :attr:`shape`.
    Tensor view(int batch, int chan, int rows, int cols,
                Storage_order new_order=Storage_order::row_major)
    {
        Tensor tview(batch, chan, rows, cols, new_order, data);
        return tview;
    }

    // Copy a block from tensor into a new tensor.
    Tensor copy(const Tensor_block &block, Storage_order order=Storage_order::row_major)
    {
      Tensor t(block.num_batch, block.num_chan,
               block.num_rows, block.num_cols, order);
      for(int batch=block.batch_min, dst_batch=0; dst_batch < block.num_batch; ++batch, ++dst_batch)
        for(int chan=block.chan_min, dst_chan=0; dst_chan < block.num_chan; ++chan, ++dst_chan)
          for(int row=block.row_min, dst_row=0; dst_row < block.num_rows; ++row, ++dst_row)
            for(int col=block.col_min, dst_col=0; dst_col < block.num_cols; ++col, ++dst_col){
              t(dst_batch, dst_chan, dst_row, dst_col) = this->operator()(batch, chan, row, col);
            }
      return t;
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
            return get_same_pad(batch, chan, row, col);
        }break;
        default:{
            return get_zero_pad(batch, chan, row, col);
        }break;
        }
    }

    //output the tensor to stdout
    void print()
    {
        for(int batch=0; batch < shape[0]; ++batch)
        {
            std::cout << "batch " << batch << std::endl;
            for(int chan=0; chan < shape[1]; ++chan)
            {
                std::cout << "  chan " << chan << std::endl;
                for(int row=0; row < shape[2]; ++row)
                {
                    std::cout << "    ";
                    for(int col=0; col < shape[3]; ++col)
                    {
                        std::cout << this->operator()(batch, chan, row, col) << " ";
                    }
                    std::cout << std::endl;
                }
            }
        }
    }

    // Number of elements in this tensor.
    int numel(){
        return std::accumulate(shape.begin(), shape.end(), 1, [](int a, int b){
            return a * b;
        });
    }

    std::array<int, 4> stride;
    std::array<int, 4> shape = {0, 0, 0, 0};
    Storage_order order;
    float *data = nullptr;
    bool own=false;

private:
    float zero=0;

    //get an element in case of zero-padding
    float &get_zero_pad(const int batch, const int chan, const int row, const int col)
    {
        if(batch < 0 or batch>=shape[0] or chan  < 0 or chan >=shape[1] or
           row   < 0 or row  >=shape[2]  or col  < 0 or col  >=shape[3])
        {
            zero=0;
            return zero;
        }
        return get_no_pad(batch, chan, row, col);
    }

    //get an element in case of same element padding
    float &get_same_pad(const int batch, const int chan, const int row, const int col)
    {
        auto new_batch = std::min(shape[0]-1, std::max(0, batch));
        auto new_chan  = std::min(shape[1]-1, std::max(0, chan));
        auto new_row   = std::min(shape[2]-1, std::max(0, row));
        auto new_col   = std::min(shape[3]-1, std::max(0, col));

        return get_no_pad(new_batch, new_chan, new_row, new_col);
    }

    //get an element considering no padding (padding done before).
    //Works with row and col-major tensors.
    float &get_no_pad(const int batch, const int chan, const int row, const int col)
    {
        int offset = col*stride[3] + row*stride[2] + chan*stride[1] + batch*stride[0];
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
    int out_ch = kern.shape[0];
    int in_ch = kern.shape[1];
    int k_h = kern.shape[2];
    int k_w = kern.shape[3];

    auto size = calc_size(src.shape[2], src.shape[3], k_h, k_w,
                          str_h, str_w, pad_h, pad_w, dil_h, dil_w);
    dst.resize(src.shape[0], out_ch, size.num_rows, size.num_cols, dst.order);

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
    int out_ch = src.shape[1];

    auto size = calc_size(src.shape[2], src.shape[3], k_h, k_w,
                          str_h, str_w, pad_h, pad_w, dil_h, dil_w);
    dst.resize(src.shape[0], out_ch, size.num_rows, size.num_cols, dst.order);

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
    int out_ch = src.shape[1];
    int kernel_size = k_h*k_w;
    auto size = calc_size(src.shape[2], src.shape[3], k_h, k_w,
                          str_h, str_w, pad_h, pad_w, dil_h, dil_w);
    dst.resize(src.shape[0], out_ch, size.num_rows, size.num_cols, dst.order);

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
  const int in_ch = 1;
  const int out_ch = 3;
  const int k_h=3, k_w=3;
  const int strd_h=2, strd_w=1;
  const int pad_h=2, pad_w=1;
  const int dil_h=2, dil_w=1;
  const int in_h = 6;
  const int in_w = 6;

  Tensor src(1, in_ch, in_h, in_w);
  for(int ch = 0; ch < in_ch; ++ch){
    float el = 0;
    for(int cur_h = 0; cur_h < in_h; ++cur_h){
      for(int cur_w = 0; cur_w < in_w; ++cur_w){
        src(0, ch, cur_h, cur_w) = ++el;
      }
    }
  }

  Tensor kernels(out_ch, in_ch, k_h, k_w);
  float kvalue = 0;
  for(int cur_out=0; cur_out<out_ch; ++cur_out){
    for(int cur_in=0; cur_in<in_ch; ++cur_in){
      for(int cur_h = 0; cur_h < k_h; ++cur_h){
        for(int cur_w=0; cur_w < k_w; ++cur_w){
            kernels(cur_out, cur_in, cur_h, cur_w) = kvalue++;
        }
      }
    }
  }

  Tensor bias(1, 1, 1, out_ch);
  for(int cur_out=0; cur_out<out_ch; ++cur_out){
    bias(0, 0, 0, cur_out) = cur_out;
  }

  Tensor dst;

  cnn2d(strd_h, strd_w,
        pad_h, pad_w, Padding_type::zero_pad,
        dil_h, dil_w,
        bias, kernels, src, dst);

  kernels.print();
  bias.print();
  src.print();
  dst.print();

  Tensor dst_pool;
  maxpool2d(2, 2,
            2, 2,
            0, 0,
            1, 1,
            dst, dst_pool);
  dst_pool.print();

  Tensor avg_pool;
  avgpool2d(2, 2,
            2, 2,
            0, 0,
            1, 1,
            dst, avg_pool);
  avg_pool.print();

  return 0;
}
#endif
