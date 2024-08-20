#include "ros/ros.h"
#include "my_package/KalmanFilter.h"
#include "my_package/kalman_filter.h"
#include <geometry_msgs/Point.h>

class KalmanService
{
public:
    KalmanService()
    {
        // Initialize the Kalman filter
        kalman_filter_ = new KalmanFilter3D();

        // Advertise the service
        service_ = nh_.advertiseService("kalman_filter", &KalmanService::handleService, this);
    }

    bool handleService(my_package::KalmanFilter::Request &req,
                       my_package::KalmanFilter::Response &res)
    {
        // Log that the service was called
        ROS_INFO("KalmanFilter service called");

        // Log the incoming request data
        ROS_INFO("Request received: x=%f, y=%f, z=%f", req.x, req.y, req.z);

        // Create a measurement from the request
        cv::Point3f measurement(req.x, req.y, req.z);

        // Update the Kalman filter
        kalman_filter_->update(measurement);

        // Predict the next state
        cv::Point3f prediction = kalman_filter_->predict();

        // Log the predicted position
        ROS_INFO("Predicted Position: x=%f, y=%f, z=%f", prediction.x, prediction.y, prediction.z);

        // Set the response
        res.x = prediction.x;
        res.y = prediction.y;
        res.z = prediction.z;

        return true;
    }

private:
    ros::NodeHandle nh_;
    ros::ServiceServer service_;
    KalmanFilter3D* kalman_filter_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "kalman_filter_node");
    KalmanService kalman_service;

    ROS_INFO("Kalman filter node started");

    ros::spin();
    return 0;
}
