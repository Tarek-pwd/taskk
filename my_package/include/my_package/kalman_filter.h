#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

#include <opencv2/core.hpp>  // Include the core OpenCV headers
#include <opencv2/video/tracking.hpp>  // Include the tracking headers for KalmanFilter

class KalmanFilter3D {
public:
    KalmanFilter3D();
    void initialize(const cv::Point3f& initial_state);
    cv::Point3f predict();
    void update(const cv::Point3f& measurement);

private:
    cv::KalmanFilter kf;
    cv::Mat measurement_mat;
};

#endif  // KALMAN_FILTER_H
