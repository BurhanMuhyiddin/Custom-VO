#include <iostream>

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>

#include <dataset_handler.hpp>

int main() {
    const std::string img_folder = "../data";

    DatasetHandler dataset_handler(img_folder, "00", false, 0);

    
    return 0;
}