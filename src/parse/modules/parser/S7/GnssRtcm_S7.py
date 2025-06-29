import struct

from modules.parser.BaseParser import BaseParser


class GnssRtcm_S7(BaseParser):

    def push_data(self, data,topic_name=None):
        u2MessageType = struct.unpack('H', data[2:4])
        u2MessageLength = struct.unpack('H', data[4:6])
        u2DataLength = struct.unpack('H', data[6:8])
        if u2MessageType[0] == 6 and u2MessageLength[0] == 514:
            self.write_bin_file(self.out_dat_path / "gnss_data.rtcm", data[8:8 + u2DataLength[0]])
