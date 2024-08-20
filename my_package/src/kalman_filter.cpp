#include "my_package/kalman_filter.h"

KalmanFilter3D::KalmanFilter3D() : kf(6, 3, 0) {
    // Transition matrix
    kf.transitionMatrix = (cv::Mat_<float>(6, 6) << 1, 0, 0, 1, 0, 0,
                                                   0, 1, 0, 0, 1, 0,
                                                   0, 0, 1, 0, 0, 1,
                                                   0, 0, 0, 1, 0, 0,
                                                   0, 0, 0, 0, 1, 0,
                                                   0, 0, 0, 0, 0, 1);
    
    // Measurement matrix
    kf.measurementMatrix = (cv::Mat_<float>(3, 6) << 1, 0, 0, 0, 0, 0,
                                                     0, 1, 0, 0, 0, 0,
                                                     0, 0, 1, 0, 0, 0);
    
    // Process noise covariance matrix
    kf.processNoiseCov = (cv::Mat_<float>(6, 6) << 1e-4, 0, 0, 0, 0, 0,
                                                    0, 1e-4, 0, 0, 0, 0,
                                                    0, 0, 1e-4, 0, 0, 0,
                                                    0, 0, 0, 1e-4, 0, 0,
                                                    0, 0, 0, 0, 1e-4, 0,
                                                    0, 0, 0, 0, 0, 1e-4);
    
    // Measurement noise covariance matrix
    kf.measurementNoiseCov = (cv::Mat_<float>(3, 3) << 1, 0, 0,
                                                       0, 1, 0,
                                                       0, 0, 1);
    
    // Initial state
    kf.statePost = cv::Mat::zeros(6, 1, CV_32F);
}

void KalmanFilter3D::initialize(const cv::Point3f& initial_state) {
    kf.statePost = (cv::Mat_<float>(6, 1) << initial_state.x, initial_state.y, initial_state.z, 0, 0, 0);
}

cv::Point3f KalmanFilter3D::predict() {
    cv::Mat prediction = kf.predict();
    return cv::Point3f(prediction.at<float>(0), prediction.at<float>(1), prediction.at<float>(2));
}

void KalmanFilter3D::update(const cv::Point3f& measurement) {
    measurement_mat = (cv::Mat_<float>(3, 1) << measurement.x, measurement.y, measurement.z);
    kf.correct(measurement_mat);
}
