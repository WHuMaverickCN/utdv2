import json
import os
import struct
from pathlib import Path

from dat_extraction_sys.modules.parser.BaseParser import BaseParser

class WheelSpeed_S7(BaseParser):
    def __init__(self, output_path, outdir=None):
        super().__init__(output_path, outdir)
        self.ws = []
        self.ws2 = []
        self.wheeldata = []
        # self.make_json()
        self.current_dat_path = None
        # self.last_time = 0.0

    def parser_data(self):
        print("parser data")
        self.make_json()
        self.write_json_file(self.out_dat_path / "wheel_speed.json", self.wheeldata)
        print(f"wheeldata size is {len(self.wheeldata)}")
        self.wheeldata.clear()

    def push_data(self, data,topic_name=None):
        u2_message_type = struct.unpack('H', data[2:4])[0]
        # ws2
        if u2_message_type == 25:
            one_frame_data = {}
            one_frame_data["timestamp"] = struct.unpack('i', data[8:12])[0] * 604800 + struct.unpack('d', data[16:24])[0] - 18.0 + 315964800
            one_frame_data["rl_wheel_velo"] = struct.unpack('d', data[24:32])[0]
            one_frame_data["rr_wheel_velo"] = struct.unpack('d', data[24:32])[0]
            self.ws2.append(one_frame_data)
        # ws
        if u2_message_type == 24:
            one_frame_data = {}
            one_frame_data["timestamp"] = struct.unpack('i', data[8:12])[0] * 604800 + struct.unpack('d', data[16:24])[0] - 18.0 + 315964800
            one_frame_data["rl_wheel_velo"] = struct.unpack('d', data[24:32])[0]
            one_frame_data["rr_wheel_velo"] = struct.unpack('d', data[24:32])[0]
            self.ws.append(one_frame_data)

    def make_json(self):
        if len(self.ws) == 0 and len(self.ws2) == 0:
            return
        if len(self.ws) == 0:
            while len(self.ws2) > 0:
                one_frame_data = {}
                one_frame_data["timestamp"] = self.ws2[0]["timestamp"]
                one_frame_data["interval"] = 10
                one_frame_data["fl_wheel_velo"] = 0xFFFF
                one_frame_data["fr_wheel_velo"] = 0xFFFF
                one_frame_data["rl_wheel_velo"] = self.ws2[0]["rl_wheel_velo"]
                one_frame_data["rr_wheel_velo"] = self.ws2[0]["rr_wheel_velo"]
                self.wheeldata.append(one_frame_data)
                self.ws2.pop(0)
                # print(f"ws2 size is {len(self.ws2)}")
            return

        if len(self.ws2) == 0:
            while len(self.ws) > 0:
                one_frame_data = {}
                one_frame_data["timestamp"] = self.ws[0]["timestamp"]
                one_frame_data["interval"] = 10
                one_frame_data["fl_wheel_velo"] = 0xFFFF
                one_frame_data["fr_wheel_velo"] = 0xFFFF
                one_frame_data["rl_wheel_velo"] = self.ws[0]["rl_wheel_velo"]
                one_frame_data["rr_wheel_velo"] = self.ws[0]["rr_wheel_velo"]
                self.wheeldata.append(one_frame_data)
                self.ws.pop(0)
                # print(f"ws1 size is {len(self.ws)}")
            return
        #第一包数据，删除未对齐部分
        if self.current_dat_path is None:
            print("current_dat_path is None")
            if self.ws2[0]["timestamp"] > self.ws[0]["timestamp"]:
                while self.ws2[0]["timestamp"] > self.ws[0]["timestamp"]:
                    self.ws.pop(0)
            if self.ws2[0]["timestamp"] < self.ws[0]["timestamp"]:
                while self.ws2[0]["timestamp"] < self.ws[0]["timestamp"]:
                    self.ws2.pop(0)
        self.current_dat_path = self.out_dat_path / "wheel_speed.json"
        print(f"reset current_dat_path : {self.current_dat_path}")
        # if self.ws[-1]["timestamp"] >= self.ws2[-1]["timestamp"]:
        #     self.last_time = self.ws[-1]["timestamp"]
        # else:
        #     self.last_time = self.ws2[-1]["timestamp"]
        #选择size较小的数据
        if len(self.ws2) <= len(self.ws):
            size_tmp = len(self.ws2)
        else:
            size_tmp = len(self.ws)
        while size_tmp != 0:
            # if self.ws[0]["timestamp"] <= self.ws2[0]["timestamp"]:
            one_frame_data = {}
            one_frame_data["timestamp"] = self.ws2[0]["timestamp"]
            one_frame_data["interval"] = 10
            one_frame_data["fl_wheel_velo"] = 0xFFFF
            one_frame_data["fr_wheel_velo"] = 0xFFFF
            one_frame_data["rl_wheel_velo"] = self.ws[0]["rl_wheel_velo"]
            one_frame_data["rr_wheel_velo"] = self.ws2[0]["rr_wheel_velo"]
            self.wheeldata.append(one_frame_data)
            self.ws.pop(0)
            self.ws2.pop(0)
            # print(f"ws2 size is {len(self.ws2)} and ws size is {len(self.ws)}")
            size_tmp = size_tmp - 1

    def write_json_file(self, file_name, file):
        with open(file_name, 'w') as res_file:
            json.dump(file, res_file)


