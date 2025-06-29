#include "map_recv_component.h"

#include <iostream>
#include "map_recv_version.h"
#include "kdata_loader.h"
#include "capilot_image_frame.h"

REGISTER_COMPONENT("map_recv", map_recv_component);

INIT_LOG(map_recv);

map_recv_component::map_recv_component() : Node("map_recv") {}

map_recv_component::~map_recv_component() {}

void map_recv_component::init() {
  KLOG_INFO("component {} version: {}", Name(), map_recv_VERSION);
  /****** please uncomment the following code as needed ******/

  bindTimerCallback(std::bind(&map_recv_component::timerCallback, this),
                    getInt("interval"));
  configInit();
  eventInit();

}

void map_recv_component::timerCallback() {
  // process all the input here...
}

void map_recv_component::configInit() {
  recev_idmap_static_channel_ = getStringVector("sub_topic")[0];
  recev_ins_channel_ = getStringVector("sub_topic")[1];
}

void map_recv_component::eventInit() {
  bindEventCallback<idmap::StaticIDMapInfo>(
      std::bind(&map_recv_component::ReceiveStaticMap, this,
                std::placeholders::_1, std::placeholders::_2),
      recev_idmap_static_channel_);

  //   bindEventCallback<localization::InsData>(
  //       std::bind(&map_recv_component::RecvInsData, this,
  //       std::placeholders::_1,
  //                 std::placeholders::_2),
  //       recev_ins_channel_);

  bindEventCallback<localization::InsData>(
      std::bind(&map_recv_component::RecvInsData, this, std::placeholders::_1,
                std::placeholders::_2),
      recev_ins_channel_);
}

void map_recv_component::ReceiveStaticMap(const idmap::StaticIDMapInfo& data,
                                          double timestamp) {
  auto aco_pos = data.anchor_pos();
  std::cout << "---static_map: " << data.ByteSizeLong()
            << ",time:" << data.circle_radius()
            << ",lane-size:" << data.lanes_size() << std::endl;
  common::SetProtoToASCIIFile(data, "./map_static_info.txt");
}

void map_recv_component::RecvInsData(const localization::InsData& ins_data,
                                     double timestamp) {
                                       std::ofstream file;
                                       double gps_velocity = 0, wheel_speed = 0, dv = 0;
  switch (ins_data.position_type())
  {
  case localization::kNone:
  case localization::kSingle:
  case localization::kPsrDiffSBAS:
  case localization::kDGPS:
  case localization::kRtkFloat:
  case localization::kRtkInt:
  case localization::kInsPos:
    std::cout << "--ins_position: " << ins_data.position_type() << std::endl;
    // std::cout << "running test" << std::endl;
    
file.open("./data.csv", std::ios::app);

gps_velocity = sqrt(ins_data.north_velocity() * ins_data.north_velocity() + ins_data.east_velocity() * ins_data.east_velocity());
wheel_speed = (ins_data.wheel_data().fl_wheel_vel() + ins_data.wheel_data().fr_wheel_vel() + ins_data.wheel_data().rl_wheel_vel() + ins_data.wheel_data().rr_wheel_vel()) / 4;
dv = abs(gps_velocity - wheel_speed);
file << std::fixed 
      << std::setprecision(0) 
      << ins_data.sec_of_week() << "," 
      << std::setprecision(0) 
      << ins_data.gps_week_number() << "," 
      << std::setprecision(2) 
      << ins_data.utc() << "," 
      << std::setprecision(0) 
      << ins_data.position_type() << "," 
      << std::setprecision(0) 
      << ins_data.numsv() << "," 
      << std::setprecision(0) 
      << ins_data.ins_status() << "," 
      << std::setprecision(2) 
      << ins_data.temperature() << "," 
      << std::setprecision(15) 
      << ins_data.latitude() << "," 
      << std::setprecision(15) 
      << ins_data.longitude() << "," 
      << std::setprecision(3) 
      << ins_data.altitude() << "," 
      << std::setprecision(3) 
      << ins_data.north_velocity() << "," 
      << std::setprecision(3) 
      << ins_data.east_velocity() << "," 
      << std::setprecision(3) 
      << ins_data.ground_velocity() << "," 
      << std::setprecision(3) 
      << ins_data.roll() << "," 
      << std::setprecision(3) 
      << ins_data.pitch() << "," 
      << std::setprecision(3) 
      << ins_data.heading() << "," 
      << std::setprecision(15) 
      << ins_data.x_angular_velocity() << "," 
      << std::setprecision(15) 
      << ins_data.y_angular_velocity() << "," 
      << std::setprecision(15) 
      << ins_data.z_angular_velocity() << "," 
      << std::setprecision(3) 
      << ins_data.x_acc() << "," 
      << std::setprecision(3) 
      << ins_data.y_acc() << "," 
      << std::setprecision(3) 
      << ins_data.z_acc() << "," 
      << std::setprecision(0) 
      << ins_data.latitude_std() << "," 
      << std::setprecision(0) 
      << ins_data.longitude_std() << "," 
      << std::setprecision(0) 
      << ins_data.altitude_std() << "," 
      << std::setprecision(0) 
      << ins_data.north_velocity_std() << "," 
      << std::setprecision(0) 
      << ins_data.east_velocity_std() << "," 
      << std::setprecision(0) 
      << ins_data.ground_velocity_std() << "," 
      << std::setprecision(3) 
      << ins_data.roll_std() << "," 
      << std::setprecision(3) 
      << ins_data.pitch_std() << "," 
      << std::setprecision(3) 
      << ins_data.heading_std() << "," 
      << std::setprecision(6) 
      << ins_data.atb_0() << "," 
      << std::setprecision(6) 
      << ins_data.atb_1() << "," 
      << std::setprecision(6) 
      << ins_data.atb_2() << "," 
      << std::setprecision(6) 
      << ins_data.q_x() << "," 
      << std::setprecision(6) 
       << ins_data.q_y() << "," 
      << std::setprecision(6) 
      << ins_data.q_z() << "," 
      << std::setprecision(6) 
      << ins_data.q_w() << "," 
      << std::setprecision(6) 
      << ins_data.wheel_data().fl_wheel_vel() << "," 
      << std::setprecision(6) 
      << ins_data.wheel_data().fr_wheel_vel() << "," 
      << std::setprecision(6) 
      << ins_data.wheel_data().rl_wheel_vel() << "," 
      << std::setprecision(6) 
      << ins_data.wheel_data().rr_wheel_vel() << "," 
      << std::setprecision(4) 
      << ins_data.wheel_data().l_wheel_factor() << "," 
      << std::setprecision(4) 
      << ins_data.wheel_data().r_wheel_factor() << "," 
      << std::setprecision(2) 
      << ins_data.header().time_stamp() << "," 
      << std::setprecision(0) 
      << ins_data.header().seq() << "," 
      << std::setprecision(0) 
      << ins_data.header().time_valid() << "," 
      << std::setprecision(0) 
      << ins_data.seq_header().timestamp_sab_vehicle() << "," 
      << std::setprecision(0) 
      << ins_data.x_angular_velocity_bias() << "," 
      << std::setprecision(0) 
      << ins_data.y_angular_velocity_bias() << "," 
      << std::setprecision(0) 
      << ins_data.z_angular_velocity_bias() << "," 
      << std::setprecision(0) 
      << ins_data.x_acc_bias() << "," 
      << std::setprecision(0) 
      << ins_data.y_acc_bias() << "," 
      << std::setprecision(0) 
      << ins_data.z_acc_bias() << "," 
      << std::setprecision(6)
      << gps_velocity << ","
      << std::setprecision(6) 
      << wheel_speed << ","
      << std::setprecision(6)  
      << dv
      << std::endl;

file.close();


{
// au-2023
// topic: sda-capilot-PubInfo-ca_adas_fc_sender-string-RawImage-fc 437

/*
  std::cout << "running test" << std::endl;
  capilot::KDataLoader kdataLoader;
  kdataLoader.LoadFile("/home/xd3/50516/map_recv/sda.dat");
  kdataLoader.PrintDataInfo();

  double timestamp = 0;
  std::string curKData;
  // std::string imgTopic = "sda-capilot-PubInfo-ca_adas_fc_sender-string-RawImage-fc";
  // std::string imgTopic = "ca_adas_fc_sender-string-RawImage-fc";sda-ca_adas_fc_sender-string-RawImage-fc
  std::string imgTopic = "sda-ca_adas_fc_sender-string-RawImage-fc";
  
  
  if(kdataLoader.GetNextData(&curKData, &imgTopic, &timestamp))
  {
    std::cout << "have next data" << std::endl;
    std::cout << "date length : " << curKData.size() << std::endl;
    cav::CapilotMultiImageFrame image_frames;
    bool ret = image_frames.decode_frame(&curKData, curKData.size());
    std::cout << "decode_frame ret : " << ret << std::endl;
    if (ret)
    {
      // 获取图像总路数
      image_frames.frame_count();
      // 获取第0路图像
      // const cav::CapilotImageFrame &frame0 = image_frames.get_frame(0);
      // const cav::CapilotImageFrame frame0 = image_frames.get_frame(0);
      cav::CapilotImageFrame frame0 = image_frames.get_frame(0);
      // 获取当前路图像帧ID
      int32_t frame0_id = frame0.frame_id();
      // 获取当前路图像时间戳
      int64_t frame0_timestamp = frame0.timestamp();
      // 获取图像数据或图像大小
      // frame0.image_data(); // imagedata数据地址
      size_t frame0_size = frame0.image_size();

      std::cout << "frame0_size : " << frame0_size << std::endl;
      std::cout << "frame0_id : " << frame0_id << std::endl;
      std::cout << "frame0_timestamp : " << frame0_timestamp << std::endl;
      std::cout << std::endl;
      std::cout << std::endl;
      std::cout << std::endl;
    }
    else 
    {
      // error 错误处理
    }   
  }*/
  /* if(kdataLoader.GetNextData(&curKData, &imgTopic, &timestamp))
  {
    std::cout << "have next data" << std::endl;
    std::cout << "date length : " << curKData.size() << std::endl;
    cav::CapilotMultiImageFrame image_frames;
    bool ret = image_frames.decode_frame(&curKData, curKData.size());
    std::cout << "decode_frame ret : " << ret << std::endl;
    if (ret)
    {
      // 获取图像总路数
      image_frames.frame_count();
      // 获取第0路图像
      // const cav::CapilotImageFrame &frame0 = image_frames.get_frame(0);
      // const cav::CapilotImageFrame frame0 = image_frames.get_frame(0);
      cav::CapilotImageFrame frame0 = image_frames.get_frame(0);
      // 获取当前路图像帧ID
      int32_t frame0_id = frame0.frame_id();
      // 获取当前路图像时间戳
      int64_t frame0_timestamp = frame0.timestamp();
      // 获取图像数据或图像大小
      // frame0.image_data(); // imagedata数据地址
      size_t frame0_size = frame0.image_size();

      std::cout << "frame0_size : " << frame0_size << std::endl;
      std::cout << "frame0_id : " << frame0_id << std::endl;
      std::cout << "frame0_timestamp : " << frame0_timestamp << std::endl;
      std::cout << std::endl;
      std::cout << std::endl;
      std::cout << std::endl;
    }
    else 
    {
      // error 错误处理
    }  
  }  */
}

  default:
    break;
  }
}
