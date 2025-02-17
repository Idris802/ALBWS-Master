#include <gazebo/common/Plugin.hh>
#include <gazebo/sensors/Sensor.hh>
#include <gazebo/sensors/RaySensor.hh>
#include <ros/ros.h>
#include <std_msgs/Bool.h>

namespace gazebo
{
  class LaserTogglePlugin : public SensorPlugin
  {
  private:
    sensors::RaySensorPtr parentSensor;
    ros::NodeHandle nh;
    ros::Subscriber toggleSub;
    bool laserActive;

  public:
    LaserTogglePlugin() : laserActive(true) {}

    void Load(sensors::SensorPtr _sensor, sdf::ElementPtr /*_sdf*/)
    {
      if (!_sensor)
      {
        ROS_ERROR("LaserTogglePlugin: No sensor pointer provided.");
        return;
      }

      parentSensor = std::dynamic_pointer_cast<sensors::RaySensor>(_sensor);
      if (!parentSensor)
      {
        ROS_ERROR("LaserTogglePlugin: Sensor is not a RaySensor.");
        return;
      }

      toggleSub = nh.subscribe<std_msgs::Bool>("/laser_pointer_toggle", 1,
        &LaserTogglePlugin::ToggleLaser, this);

      ROS_INFO("LaserTogglePlugin loaded successfully.");
    }

    void ToggleLaser(const std_msgs::Bool::ConstPtr &msg)
    {
        laserActive = msg->data;

        if (!laserActive) 
        {
            parentSensor->SetActive(false);
            ROS_INFO("Laser Deactivated");
        }
        else 
        {
            parentSensor->SetActive(true);
            ROS_INFO("Laser Activated");
        }

        // Force a sensor update
        parentSensor->Update(true);
    }

  };

  GZ_REGISTER_SENSOR_PLUGIN(LaserTogglePlugin)
}
