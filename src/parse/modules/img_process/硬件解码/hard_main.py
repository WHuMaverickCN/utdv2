import os
import struct
from pathlib import Path
from potime import RunTime
import yaml
import shutil

from hard_image import get_image

"""
- struct: 
    - https://blog.51cto.com/u_16175495/6664877
    - https://blog.51cto.com/u_16213640/8792450
"""


def read_header(file):
    """
    读取文件头信息
    :param file:
    :return:
    """
    byte_data = file.read(214)
    proto_length = struct.unpack('N', byte_data[4:12])
    comment_length = struct.unpack('h', byte_data[13:15])
    if comment_length[0] > 0 & proto_length[0] > 0:
        file.read(comment_length[0])
        file.read(proto_length[0])
    return file, byte_data  # 返回文件头内容，拼接成最后的文件


def read_topic(file):
    """
    读取前116个字节，包含了topic的信息
    :param file:
    :return:
    """
    byte_data = file.read(100)
    topic_name = byte_data.decode()

    topic_size_bin = file.read(8)
    topic_size = struct.unpack('N', topic_size_bin)

    topic_timestamp_bin = file.read(8)
    topic_timestamp = struct.unpack('d', topic_timestamp_bin)
    return file, topic_name, topic_size, topic_timestamp


@RunTime
def split_topic(output_path, dat_file):
    dat_path = Path(output_path) / Path(dat_file).stem
    if dat_path.exists():
        shutil.rmtree(dat_path)
    os.makedirs(dat_path)
    with open(dat_file, 'rb') as file:
        file_size = os.stat(dat_file).st_size
        if file_size < 116:
            pass
        else:
            file, header_bin = read_header(file)
            # # 写入dat的header
            with open('../../parser/config/settings.yaml', 'r', encoding='utf8') as f:
                s_y = yaml.load(f.read(), Loader=yaml.FullLoader)
                topic_list = s_y['Pasers'][0]['TopicNames']  # DatStatus 文件夹路径
            # 取出topic内容
            while True:
                file, topic_name, topic_size, topic_timestamp = read_topic(file)
                if topic_name.strip('\x00') in topic_list:
                    topic_content = file.read(topic_size[0])
                    get_image(topic_content, dat_path / topic_name.strip('\x00'), topic_timestamp)
                else:
                    file.seek(topic_size[0], 1)  # 跳过这一帧的topic长度
                if file.tell() >= file_size:  # 超过文件大小，退出
                    break


if __name__ == '__main__':
    output_path = r'./output'
    dat_file = r'C:\work\dat\all\4\S7_003494_2024-01-20_19-25-00.dat'
    split_topic(output_path, dat_file)
