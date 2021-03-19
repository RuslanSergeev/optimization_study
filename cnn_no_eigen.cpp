#include <iostream>
#include <array>
#include <functional>
#include <iostream>
#include <limits>
using namespace std;

enum class padding_type {
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
  Tensor(int num_batch=0, int num_chan=0, int num_rows=0, int num_cols=0)
  {
    resize(num_batch, num_chan, num_rows, num_cols);
  }

  virtual ~Tensor()
  {
    if(own)
    {
      delete[] data;
    }
  }

  void resize(int num_batch, int num_chan, int num_rows, int num_cols)
  {
    int new_size = num_cols*num_rows*num_chan*num_batch;
    if(new_size!=size and own)
    {
      delete[] data;
    }
    size = new_size;
    if(size > 0)
    {
      data = new float[size];
      own  = true;
    }
    stride[3] = 1;
    stride[2] = stride[3]*num_cols;
    stride[1] = stride[2]*num_rows;
    stride[0] = stride[1]*num_chan;
    b.num_batch = num_batch;
    b.num_chan = num_chan;
    b.num_rows = num_rows;
    b.num_cols = num_cols;
  }

  Tensor copy(const Tensor_block &block)
  {
    Tensor t(block.num_batch, block.num_chan, block.num_rows, block.num_cols);
    for(int batch=block.batch_min, dst_batch=0; dst_batch < block.num_batch; ++batch, ++dst_batch)
    {
      for(int chan=block.chan_min, dst_chan=0; dst_chan < block.num_chan; ++chan, ++dst_chan)
      {
        for(int row=block.row_min, dst_row=0; dst_row < block.num_rows; ++row, ++dst_row)
        {
          for(int col=block.col_min, dst_col=0; dst_col < block.num_cols; ++col, ++dst_col)
          {
            t(dst_batch, dst_chan, dst_row, dst_col) = this->operator()(batch, chan, row, col);
          }
        }
      }
    }
    return t;
  }

  Tensor view(const Tensor_block &block)
  {
    Tensor t(0, 0, 0, 0);
    t.b = block;
    t.stride = stride;
    t.data = data;
    t.size = block.size();
    t.own = false;

    return t;
  }

  float &operator()(const int batch, const int chan, const int row, const int col, padding_type pad_type=padding_type::zero_pad)
  {
    int mb = batch+b.batch_min;
    int mc = chan+b.chan_min;
    int mrow = row+b.row_min;
    int mcol = col+b.col_min;
    switch(pad_type)
    {
      case padding_type::zero_pad:
      {
        return get_zero_pad(mb, mc, mrow, mcol);
      } break;
      case padding_type::same_pad:
      {
        return get_same_pad(mb, mc, mrow, mcol);
      }break;
    }
  }

  void print()
  {
    for(int batch=0; batch < b.num_batch; ++batch)
    {
      std::cout << "batch " << batch << std::endl;
      for(int chan=0; chan < b.num_chan; ++chan)
      {
        std::cout << "  chan " << chan << std::endl;
        for(int row=0; row < b.num_rows; ++row)
        {
          std::cout << "    ";
          for(int col=0; col < b.num_cols; ++col)
          {
            std::cout << this->operator()(batch, chan, row, col) << " ";
          }
          std::cout << std::endl;
        }
      }
    }
  }

  Tensor_block b;
  std::array<int, 4> stride;
  float *data = nullptr;
  int size = 0;
  bool own=false;

private:
  float zero=0;

  float &get_zero_pad(const int batch, const int chan, const int row, const int col)
  {
    if(batch < b.batch_min or batch>=b.batch_min+b.num_batch or
       chan  < b.chan_min or chan >=b.chan_min+b.num_chan  or
       row   < b.row_min or row  >=b.row_min+b.num_rows  or
       col   < b.col_min or col  >=b.col_min+b.num_cols)
    {
      zero=0;
      return zero;
    }
    return get_no_pad(batch, chan, row, col);
  }

  float &get_same_pad(const int batch, const int chan, const int row, const int col)
  {
    auto new_batch = std::min(b.batch_min+b.num_batch-1, std::max(b.batch_min, batch));
    auto new_chan  = std::min(b.chan_min+b.num_chan-1,  std::max(b.chan_min, chan));
    auto new_row   = std::min(b.row_min+b.num_rows-1,  std::max(b.row_min, row));
    auto new_col   = std::min(b.col_min+b.num_cols-1,  std::max(b.col_min, col));

    return get_no_pad(new_batch, new_chan, new_row, new_col);
  }

  float &get_no_pad(const int batch, const int chan, const int row, const int col)
  {
    int offset = col*stride[3] + row*stride[2] + chan*stride[1] + batch*stride[0];
    return data[offset];
  }
};


Tensor_block calc_size(const int h_in, const int w_in,
               const int k_h, const int k_w,
               const int str_h, const int str_w,
               const int pad_h, const int pad_w,
               const int dil_h, const int dil_w)
{
  Tensor_block o;
  o.num_rows = int((h_in+2*pad_h-dil_h*(k_h-1)-1)/str_h+1);
  o.num_cols = int((w_in+2*pad_w-dil_w*(k_w-1)-1)/str_w+1);
  return o;
}

void cnn2d(const int str_h, const int str_w,
           const int pad_h, const int pad_w, padding_type ptype,
           const int dil_h, const int dil_w,
           Tensor &bias,
           Tensor &kern,
           Tensor &src,
           Tensor &dst)
{
  int out_ch = kern.b.num_batch;
  int in_ch = kern.b.num_chan;
  int k_h = kern.b.num_rows;
  int k_w = kern.b.num_cols;

  auto size = calc_size(src.b.num_rows, src.b.num_cols, k_h, k_w,
                        str_h, str_w, pad_h, pad_w, dil_h, dil_w);
  if(not dst.size){
    dst.resize(src.b.num_batch, out_ch, size.num_rows, size.num_cols);
  }

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
  int out_ch = src.b.num_chan;

  auto size = calc_size(src.b.num_rows, src.b.num_cols, k_h, k_w,
                        str_h, str_w, pad_h, pad_w, dil_h, dil_w);
  if(not dst.size){
    dst.resize(src.b.num_batch, out_ch, size.num_rows, size.num_cols);
  }

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
  int out_ch = src.b.num_chan;
  int kernel_size = k_h*k_w;
  auto size = calc_size(src.b.num_rows, src.b.num_cols, k_h, k_w,
                        str_h, str_w, pad_h, pad_w, dil_h, dil_w);
  if(not dst.size){
    dst.resize(src.b.num_batch, out_ch, size.num_rows, size.num_cols);
  }

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
        pad_h, pad_w, padding_type::zero_pad,
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

  // std::cout << std::endl;
  // Tensor_block b;
  // b.num_batch = 1;
  // b.num_chan = 2;
  // b.num_rows = 3;
  // b.num_cols = 3;
  // Tensor t(b.num_batch, b.num_chan, b.num_rows, b.num_cols);
  // float buf[18];
  // for(int i=0; i<18; ++i)
  // {
  //   t.data[i] = float(i+1);
  // }
  // t.print();
  //
  // Tensor_block block;
  // block.batch_min = 0; block.num_batch=b.num_batch;
  // block.chan_min = 0; block.num_chan=b.num_chan;
  // block.row_min = 0; block.num_rows=2;
  // block.col_min = 0; block.num_cols=2;
  // auto tc = t.copy(block);
  // tc.print();
  //
  // for(int row=-1; row<3; ++row)
  // {
  //   for(int col=-1; col<3; ++col)
  //   {
  //     std::cout << tc(0, 0, row, col, padding_type::same_pad) << " ";
  //   }
  //   std::cout << std::endl;
  // }
  //
  // Tensor_block bv;
  // bv.batch_min = 0; bv.num_batch=b.num_batch;
  // bv.chan_min = 0; bv.num_chan=b.num_chan;
  // bv.row_min = 1; bv.num_rows=1;
  // bv.col_min = 1; bv.num_cols=1;
  // auto tv = t.view(bv);
  // tv.print();
  //
  // for(int row=-1; row<2; ++row)
  // {
  //   for(int col=-1; col<2; ++col)
  //   {
  //     std::cout << tv(0, 0, row, col, padding_type::same_pad) << " ";
  //   }
  //   std::cout << std::endl;
  // }
  //
  // for(int row=-1; row<2; ++row)
  // {
  //   for(int col=-1; col<2; ++col)
  //   {
  //     std::cout << tv(0, 1, row, col, padding_type::same_pad) << " ";
  //   }
  //   std::cout << std::endl;
  // }

  return 0;
}
