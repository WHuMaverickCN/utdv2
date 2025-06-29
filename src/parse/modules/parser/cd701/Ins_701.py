from modules.proto.proto_py import ins_data_pb2

from ..BaseParser import BaseParser


class Ins_701(BaseParser):
    def __init__(self, output_path, outdir=None):
        super().__init__(output_path, outdir)
        self.ins_struct = None
        self.imu = []
        self.pos_imu = []
        self.wheel = []

    def parser_data(self):
        self.write_json_file(self.out_dat_path / "pos_imu_data.json", self.pos_imu)
        self.write_json_file(self.out_dat_path / "imu_data.json", self.imu)
        self.write_json_file(self.out_dat_path / "wheel_speed_data.json", self.wheel)

    def push_data(self, data,topic_name=None):
        self.ins_struct = ins_data_pb2.InsData()
        ins_res = self.ins_struct.ParseFromString(data)
        if ins_res:
            self.ImuDataConvertor()
            self.PosImuConvertor()
            self.WheelDataConvertor()
            return True, ins_res

    def ImuDataConvertor(self):
        one_frame_data = {}
        one_frame_data["interval"] = 10
        one_frame_data["temperature"] = self.ins_struct.temperature
        one_frame_data["valueY"] = self.ins_struct.y_angular_velocity
        one_frame_data["valueX"] = self.ins_struct.x_angular_velocity
        one_frame_data["valueZ"] = self.ins_struct.z_angular_velocity
        one_frame_data["valueAxis"] = 3
        one_frame_data["acceX"] = self.ins_struct.x_acc
        one_frame_data["acceY"] = self.ins_struct.y_acc
        one_frame_data["acceZ"] = self.ins_struct.z_acc
        one_frame_data["acceAxis"] = 3
        one_frame_data["tickTime"] = self.ins_struct.device_time
        self.imu.append(one_frame_data)

    def get_utc(self):
        return self.ins_struct.utc

    def get_cts(self):
        return self.ins_struct.device_time

    def PosImuConvertor(self):
        one_frame_data = {}

        one_frame_data["timestamp"] = self.ins_struct.device_time
        one_frame_data["angular_velocity"] = []
        one_frame_data["angular_velocity"].append(self.ins_struct.x_angular_velocity)
        one_frame_data["angular_velocity"].append(self.ins_struct.y_angular_velocity)
        one_frame_data["angular_velocity"].append(self.ins_struct.z_angular_velocity)
        one_frame_data["acc"] = []
        one_frame_data["acc"].append(self.ins_struct.x_acc)
        one_frame_data["acc"].append(self.ins_struct.y_acc)
        one_frame_data["acc"].append(self.ins_struct.z_acc)
        one_frame_data["gps_loc"] = []
        one_frame_data["gps_loc"].append(self.ins_struct.latitude)
        one_frame_data["gps_loc"].append(self.ins_struct.longitude)
        one_frame_data["gps_loc"].append(self.ins_struct.altitude)
        one_frame_data["neg_velocity"] = []
        one_frame_data["neg_velocity"].append(self.ins_struct.east_velocity)
        one_frame_data["neg_velocity"].append(self.ins_struct.north_velocity_std)
        one_frame_data["neg_velocity"].append(self.ins_struct.ground_velocity)
        one_frame_data["rph"] = []
        one_frame_data["rph"].append(self.ins_struct.roll)
        one_frame_data["rph"].append(self.ins_struct.pitch)
        one_frame_data["rph"].append(self.ins_struct.heading)
        self.pos_imu.append(one_frame_data)

    def WheelDataConvertor(self):
        current_json_content = {}
        current_json_content["timestamp"] = self.ins_struct.device_time
        current_json_content["interval"] = 10
        current_json_content["fl_wheel_velo"] = self.ins_struct.wheel_data.fl_wheel_vel
        current_json_content["fr_wheel_velo"] = self.ins_struct.wheel_data.fr_wheel_vel
        current_json_content["rl_wheel_velo"] = self.ins_struct.wheel_data.rl_wheel_vel
        current_json_content["rr_wheel_velo"] = self.ins_struct.wheel_data.rr_wheel_vel
        self.wheel.append(current_json_content)
