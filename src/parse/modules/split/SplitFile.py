import json
import multiprocessing
import os
import shutil
import struct
from pathlib import Path

from potime import RunTime

from modules.parser.config.settings import *


class SplitFile:
    def __init__(self, setting_yaml, output_path=None):
        # if output_path == None:
        #     output_path = Path(setting_yaml[Settings.output_key_name]).absolute()
            # output_path = Path(setting_yaml[Settings.output_key_name]).absolute()
        self.output_path = Path('insdata').absolute()  # 输出文件夹
        # self.output_path = output_path
        self.setting_yaml = setting_yaml
        self.parser_key_name = Settings.parser_key_name
        self.topic_key_name = Settings.topic_key_name
        self.processes = 10

    def read_header(self, file):
        """
        读取文件头信息
        :param file:
        :return:
        :struct:
            - https://blog.51cto.com/u_16175495/6664877
            - https://blog.51cto.com/u_16213640/8792450
        """
        byte_data = file.read(214)
        proto_length = struct.unpack('Q', byte_data[4:12])
        comment_length = struct.unpack('h', byte_data[12:14])
        print('comment_length[0]', comment_length[0])
        print('proto_length[0]', proto_length[0])
        file.read(proto_length[0])
        file.read(comment_length[0])
        return file, byte_data  # 返回文件头内容，拼接成最后的文件

    def read_topic(self, file):
        """
        读取前116个字节，包含了topic的信息
        :param file:
        :return:
        """
        byte_data = file.read(100)
        topic_name = byte_data.split(b'\x00')[0].decode()

        topic_size_bin = file.read(8)
        topic_size = struct.unpack('N', topic_size_bin)

        topic_timestamp_bin = file.read(8)
        topic_timestamp = struct.unpack('d', topic_timestamp_bin)
        return file, topic_name, topic_size, topic_timestamp

    def write_topic_file(self, file_name, topic_offset, topic_size, topic_timestamp, topic_content):
        """
        输出topic文件，分为索引和内容2个文件
        :param file_name:
        :param topic_size:
        :param topic_timestamp:
        :param topic_content:
        :return:
        """
        with open(f"{file_name}{DAT_INDEX}{DAT_SUFFIX}", 'ab') as out_file:
            # out_file.write(topic_name.encode())
            out_file.write(struct.pack('N', topic_offset))
            out_file.write(struct.pack('N', topic_size[0]))
            out_file.write(struct.pack('d', topic_timestamp[0]))
        with open(f"{file_name}{DAT_SUFFIX}", 'ab') as out_file:
            out_file.write(topic_content)

    def write_file_header(self, file_name, header_bin):
        with open(f"{file_name}{DAT_SUFFIX}", 'ab') as out_file:
            out_file.write(header_bin)

    # @RunTime
    def split_dat(self, dat_file: list):
        print(self.output_path)
        print(dat_file)
        dat_path = Path(self.output_path) / Path(dat_file).stem
        if dat_path.exists():
            shutil.rmtree(dat_path)
        os.makedirs(dat_path)
        with open(dat_file, 'rb') as file:
            file_size = os.stat(dat_file).st_size
            if file_size < 116:
                pass
            else:
                file, header_bin = self.read_header(file)
                # 写入dat的header
                topic_list = []
                # with open(setting_path, 'r', encoding='utf8') as f:
                #     setting_yaml = yaml.load(f.read(), Loader=yaml.FullLoader)

                for i in range(len(self.setting_yaml[self.parser_key_name])):  # 获取所有Pasers
                    # print(self.setting_yaml[self.parser_key_name][i][self.topic_key_name])
                    topic_list = topic_list + self.setting_yaml[self.parser_key_name][i][
                        self.topic_key_name]  # DatStatus 文件夹路径
                    # for topic in topic_list:写入文件头
                    #     self.write_file_header(dat_path / topic, header_bin)
                # 取出topic内容
                # print(topic_list)
                topic_offset_counter = {}
                while True:
                    file, topic_name, topic_size, topic_timestamp = self.read_topic(file)
                    if topic_name in topic_list:
                        topic_offset = topic_offset_counter.get(topic_name, 0)
                        topic_content = file.read(topic_size[0])
                        self.write_topic_file(dat_path / topic_name, topic_offset, topic_size,
                                              topic_timestamp, topic_content)
                        topic_offset += topic_size[0]
                        topic_offset_counter[topic_name] = topic_offset

                    else:
                        file.seek(topic_size[0], 1)  # 跳过这一帧的topic长度
                    if file.tell() >= file_size:  # 超过文件大小，退出
                        break
                if not topic_offset_counter:
                    return
        return [dat_file]

    def multi_split(self, dat_list: list, processes: int):
        if len(dat_list) <= processes:  # 进程数少于数据文件数
            processes = len(dat_list)  # 进程数等于数据文件数
        print(f'processes:{processes}')
        with multiprocessing.Pool(processes=processes) as pool:
            # 向进程池添加任务
            for d_f in dat_list:
                pool.apply_async(self.split_dat, (d_f,))
            # 等待所有进程完成
            pool.close()
            pool.join()

        print("All processes are done.")

    @RunTime
    def split_dir(self, dir_path):
        files = [file.absolute() for file in Path(dir_path).absolute().iterdir()]
        self.multi_split(files, self.processes)
        return files

    @RunTime
    def split_task(self, task_json_path):
        """
        解析task文件
        :param task_json_path:
        :param processes:
        :return:
        """
        with open(task_json_path, 'r', encoding='utf-8') as task_json:
            task_json_file_content = task_json.read()
            task_json_content = json.loads(task_json_file_content)
            dat_files = task_json_content["dat_files"]
            self.multi_split(dat_files, self.processes)


if __name__ == '__main__':
    output_path = r'./output'
    # dat_file = r'/mnt/c/work/dat/all/4/S7_003494_2024-01-20_19-25-00.dat'
    # dat_file = r'C:\work\dat\all\4\S7_003494_2024-01-20_19-25-00.dat'
    # dat_file = r'C:\work\dat\all\test\cd701\CD701_000052__2024-03-19_13-34-10.dat'
    dat_file = r'E:\000052车\0325_route2-8\CD701_000052__2024-03-25_10-35-06.dat'
    sf = SplitFile(Settings.setting_yaml)
    sf.split_dat(dat_file)
