#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

#include "src/layer.h"
#include "src/layer/conv_Host.h"
#include "src/layer/conv_Device.h"
#include "src/layer/fully_connected.h"
#include "src/layer/ave_pooling.h"
#include "src/layer/max_pooling.h"
#include "src/layer/relu.h"
#include "src/layer/sigmoid.h"
#include "src/layer/softmax.h"
#include "src/loss.h"
#include "src/loss/mse_loss.h"
#include "src/loss/cross_entropy_loss.h"
#include "src/mnist.h"
#include "src/network.h"
#include "src/optimizer.h"
#include "src/optimizer/sgd.h"
#include "src/layer/gpu_utils.h"
#include "dnnNetwork.h"


int main()
{
    // Check GPU
    GPU_Utils gpu_utils;
    gpu_utils.printDeviceInfo();    
    
    // Load data
    MNIST dataset("./data/fashion/");
    dataset.read();
    int n_train = dataset.train_data.cols();
    int dim_in = dataset.train_data.rows();
    std::cout << "mnist train number: " << n_train << std::endl;
    std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;
    
    float accuracy = 0.0;
    std::cout << "------------------------------" << std::endl;
	
    // Host - CPU Network
    std::cout << "Test: CPU Network" << std::endl;
    Network dnn1 = dnnNetwork_CPU();
    dnn1.load_parameters("./weghts/fashion_weights.bin");
    dnn1.forward(dataset.test_data);
    accuracy = compute_accuracy(dnn1.output(), dataset.test_labels);
    std::cout << "test accuracy: " << accuracy << std::endl;
    std::cout << "------------------------------" << std::endl;

    // Device - GPU Network
	std::cout << "Test: GPU Network" << std::endl;
    Network dnn2 = dnnNetwork_GPU();
    dnn2.load_parameters("./weights/fashion_weights.bin");
    dnn2.forward(dataset.test_data);
    accuracy = compute_accuracy(dnn2.output(), dataset.test_labels);
    std::cout << "test accuracy: " << accuracy << std::endl;
    
    return 0;
}