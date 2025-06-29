import os
import struct

from modules.parser.config.settings import *


class ReadDat():
    def __init__(self, dat_path, parser):
        self.dat_path = dat_path
        self.parser = parser

    def push_each_frame(self, topic_name):
        """
        解析索引文件
        :return:
        """
        push_status = False
        dat_index_list = []
        topic_index_path = f'{self.dat_path / topic_name}{DAT_INDEX}{DAT_SUFFIX}'
        with open(topic_index_path, 'rb') as dat_index_file:
            try:
                while dat_index_file.tell() < os.stat(topic_index_path).st_size:
                    dat_index_list.append((struct.unpack('N', dat_index_file.read(8)),
                                           struct.unpack('N', dat_index_file.read(8)),
                                           struct.unpack('d', dat_index_file.read(8)),))
            except:  # 索引文件读错了，直到读完
                return push_status
        if len(dat_index_list) <= 0:
            return push_status
        push_status = self.push_main(topic_name, dat_index_list)
        return push_status

    def push_main(self, topic_name, dat_index_list):
        read_dat_file_status = True
        with open(f'{self.dat_path / topic_name}{DAT_SUFFIX}', 'rb') as dat_file:
            for offset, size, timestamp in dat_index_list:
                dat_file.seek(offset[0])  # 从offset开始读
                data = dat_file.read(size[0])
                self.parser.push_data(data, topic_name)
            # try:
            #     for offset, size, timestamp in dat_index_list:
            #         dat_file.seek(offset[0])  # 从offset开始读
            #         data = dat_file.read(size[0])
            #         self.parser.push_data(data, topic_name)
            # except:
            #     read_dat_file_status = False
        return read_dat_file_status
