import struct

from ..BaseParser import BaseParser
from ..config.settings import DAT_SUFFIX


class Imu_S7(BaseParser):
    def __init__(self, output_path, outdir=None):
        super().__init__(output_path, outdir)
        self.imu = []

    def parser_data(self):
        self.write_json_file(self.out_dat_path / "imu.json", self.imu)

    def push_data(self, data,topic_name=None):
        one_frame_data = {}
        one_frame_data["interval"] = 10
        one_frame_data["temperature"] = 0xffff
        one_frame_data["valueX"] = struct.unpack('d', data[24:32])[0]
        one_frame_data["valueY"] = struct.unpack('d', data[32:40])[0]
        one_frame_data["valueZ"] = struct.unpack('d', data[40:48])[0]
        one_frame_data["valueAxis"] = 3
        one_frame_data["acceX"] = struct.unpack('d', data[48:56])[0]
        one_frame_data["acceY"] = struct.unpack('d', data[56:64])[0]
        one_frame_data["acceZ"] = struct.unpack('d', data[64:72])[0]
        one_frame_data["acceAxis"] = 3
        one_frame_data["tickTime"] = struct.unpack('i', data[8:12])[0] * 604800 + struct.unpack('d', data[16:24])[
            0] - 18.0 + 315964800
        self.imu.append(one_frame_data)
