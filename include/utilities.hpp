#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <algorithm>
#include <string>
#include <fstream>

#include <opencv2/opencv.hpp>

class Error {
public:
    void SetMSE(double mse) { mse_ = mse; }
    void SetMAE(double mae) { mae_ = mae; }
    void SetRMSE(double rmse) { rmse_ = rmse; }
    double GetMSE() const { return mse_; }
    double GetMAE() const { return mae_; }
    double GetRMSE() const { return rmse_; }
private:
    double mse_;
    double mae_;
    double rmse_;
};

std::string to_uppercase(const std::string &inp_string);

double get_mae(const std::vector<cv::Mat> &ground_truth, const std::vector<cv::Mat> &estimated);

double get_mse(const std::vector<cv::Mat> &ground_truth, const std::vector<cv::Mat> &estimated);

void save_trajectory(const std::vector<cv::Mat> &trajectory, const std::string &file_path);

#endif // UTILITIES_HPP