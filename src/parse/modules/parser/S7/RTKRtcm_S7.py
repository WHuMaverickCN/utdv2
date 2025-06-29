from ..BaseParser import BaseParser


class RTKRtcm_S7(BaseParser):

    def push_data(self, data,topic_name=None):
        self.write_bin_file(self.out_dat_path / "rtk_rtcm_s7.rtcm", data)
