#ifndef SRC_LAYER_CONV_DEVICE_FORWARD_H
#define SRC_LAYER_CONV_DEVICE_FORWARD_H
#include "./gpu_utils.h"

class ConvForward
{
    public:
    void get_device_properties();
    void conv_forward_gpu(float *output_data, const float *input_data, const float *weight_data,
                          const int num_samples, const int output_channel, const int input_channel,
                          const int height_in, const int width_in, const int kernel_height);
};

#endif