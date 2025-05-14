#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <ros/ros.h>
#include <std_msgs/Bool.h>

namespace gazebo
{
  class LaserPointerPlugin : public ModelPlugin
  {
  private:
    // ...
    std::string visualName;
    
  public:
    // Constructor
    LaserPointerPlugin() : ModelPlugin() {}
    
    // Called when the plugin is loaded into Gazebo
    void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf) override
    {
      // Store the pointer to the model
      this->model = _model;
      this->modelName = _model->GetName();
    
      // Get the link name for the laser beam from SDF, or use default
      if (_sdf->HasElement("laser_link_name")) {
        this->laserLinkName = _sdf->Get<std::string>("laser_link_name");
      } else {
        this->laserLinkName = "laser_beam";  // default link name
      }
      // Check that the link exists in the model
      if (!this->model->GetLink(this->laserLinkName)) {
        gzerr << "[LaserPointerPlugin] Link '" << this->laserLinkName 
              << "' not found in model, plugin will not toggle laser.\n";
      }
      
      // Ensure ROS node is initialized (Gazebo ROS must be launched)
      if (!ros::isInitialized()) {
        ROS_FATAL_STREAM("ROS node for Gazebo not initialized. "
                         << "Make sure to launch Gazebo with ROS (e.g., empty_world.launch).");
        return;
      }
      // Create ROS node handle (in plugin namespace or global)
      this->rosNode.reset(new ros::NodeHandle(""));  // empty namespace = global
      // Subscribe to the /set_laser_pointer topic (std_msgs/Bool)
      this->rosSub = this->rosNode->subscribe<std_msgs::Bool>(
          "/set_laser_pointer", 1, 
          boost::bind(&LaserPointerPlugin::OnRosMsg, this, _1));
      ROS_INFO_STREAM("LaserPointerPlugin subscribed to /set_laser_pointer");
      ROS_INFO_STREAM("Model Name: " << this->modelName);
      ROS_INFO_STREAM("Laser Link Name: " << this->laserLinkName);
      ROS_INFO_STREAM("Full visual name: " << this->modelName + "::" + this->laserLinkName + "::laser_beam_visual");

      // Initialize Gazebo transport node and publisher for visual messages
      this->gzNode = transport::NodePtr(new transport::Node());
      this->gzNode->Init(this->model->GetWorld()->Name());
      this->gzVisPub = this->gzNode->Advertise<gazebo::msgs::Visual>("~/visual");
      this->gzVisPub->WaitForConnection();  // wait for Gazebo GUI to subscribe
      
      std::string beamVisualName = "robot::wrist_3_link::wrist_3_link_fixed_joint_lump__laser_beam_visual_visual_2"; // this->modelName + "::" + this->laserLinkName + "::laser_beam_visual";
      gazebo::msgs::Visual initMsg;
      initMsg.set_name(beamVisualName);
      initMsg.set_parent_name(this->modelName + "::" + this->laserLinkName);
      initMsg.set_visible(false);
      initMsg.set_transparency(1.0);
      this->gzVisPub->Publish(initMsg);
    }
    
  private:
    // ROS callback when a Bool message is received
    void OnRosMsg(const std_msgs::Bool::ConstPtr &msg)
    {
      std::string beamVisualName = "robot::wrist_3_link::wrist_3_link_fixed_joint_lump__laser_beam_visual_visual_2"; // this->modelName + "::" + this->laserLinkName + "::laser_beam_visual";
      bool turnOn = msg->data;
      // Prepare Visual message
      gazebo::msgs::Visual visMsg;
      visMsg.set_name(beamVisualName);
      visMsg.set_parent_name(this->modelName + "::" + this->laserLinkName);
      if (turnOn) {
        visMsg.set_visible(true);
        visMsg.set_transparency(0.0);    // fully opaque
      } else {
        visMsg.set_visible(false);
        visMsg.set_transparency(1.0);    // fully transparent
      }
      this->gzVisPub->Publish(visMsg);   // publish to Gazebo to update visual
      ROS_INFO_STREAM("Laser beam toggled " << (turnOn ? "ON" : "OFF"));
    }
    
    // Pointer to the model
    physics::ModelPtr model;
    std::string modelName;
    std::string laserLinkName;
    // Gazebo transport for visuals
    transport::NodePtr    gzNode;
    transport::PublisherPtr gzVisPub;
    // ROS subscriber
    std::unique_ptr<ros::NodeHandle> rosNode;
    ros::Subscriber rosSub;
  };
  
  // Register this plugin with the simulator
  GZ_REGISTER_MODEL_PLUGIN(LaserPointerPlugin)
}
