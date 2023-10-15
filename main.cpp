#include <iostream>

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>

int main() {
    cv::Mat img = cv::Mat::zeros(200, 400, CV_8UC3);

    // Put "Hello, OpenCV!" text on the image
    cv::putText(img, "Hello, OpenCV!", cv::Point(50, 100), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0), 2);

    // Show the image in a window
    cv::imshow("Hello OpenCV", img);

    // Wait for a key event and close the window
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}