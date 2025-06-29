import struct

from ..BaseParser import BaseParser
from ..config.settings import DAT_SUFFIX

class Gnss_S7(BaseParser):
    def __init__(self, output_path, outdir=None):
        super().__init__(output_path, outdir)
        self.gnss = []

    def parser_data(self):
        self.write_json_file(self.out_dat_path / "gnss.json", self.gnss)

    def push_data(self, data,topic_name=None):
        one_frame_data = {}
        one_frame_data["timestamp"] = struct.unpack('i', data[8:12])[0] * 604800 + struct.unpack('d', data[16:24])[
            0] - 18.0 + 315964800
        one_frame_data["latitude"] = struct.unpack('d', data[24:32])[0]
        one_frame_data["longitude"] = struct.unpack('d', data[32:40])[0]
        one_frame_data["height"] = struct.unpack('d', data[40:48])[0]
        one_frame_data["latitude_sdd"] = struct.unpack('d', data[48:56])[0]
        one_frame_data["longitude_sdd"] = struct.unpack('d', data[56:64])[0]
        one_frame_data["height_sdd"] = struct.unpack('d', data[64:72])[0]
        one_frame_data["latitude_speed"] = struct.unpack('d', data[72:80])[0]
        one_frame_data["longitude_speed"] = struct.unpack('d', data[80:88])[0]
        one_frame_data["height_speed"] = struct.unpack('d', data[88:96])[0]
        one_frame_data["latitude_speed_sdd"] = struct.unpack('d', data[96:104])[0]
        one_frame_data["longitude_speed_sdd"] = struct.unpack('d', data[104:112])[0]
        one_frame_data["height_speed_sdd"] = struct.unpack('d', data[112:120])[0]
        one_frame_data["quality"] = struct.unpack('i', data[120:124])[0]
        one_frame_data["solve_type"] = struct.unpack('i', data[124:128])[0]
        self.gnss.append(one_frame_data)
