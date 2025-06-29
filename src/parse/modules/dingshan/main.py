import shutil
import struct

import yaml
from potime import RunTime

from modules.proto.proto_py import encode_srv_pb2
from image import *
import json

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
    proto_length = struct.unpack('Q', byte_data[4:12])
    comment_length = struct.unpack('h', byte_data[12:14])
    print('comment_length[0]', comment_length[0])
    print('proto_length[0]', proto_length[0])
    file.read(proto_length[0])
    file.read(comment_length[0])
    return file, byte_data  # 返回文件头内容，拼接成最后的文件


def read_topic(file):
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
    return file, topic_name, topic_size, topic_timestamp[0]


@RunTime
def parser_topic(output_path, dat_file):
    dat_path = Path(output_path) / Path(dat_file).stem
    if dat_path.exists():
        shutil.rmtree(dat_path)
    os.makedirs(dat_path)

    output_path = str(dat_path)
    with open(dat_file, 'rb') as file:
        file_size = os.stat(dat_file).st_size
        if file_size < 116:
            pass
        else:
            file, header_bin = read_header(file)
            # # 写入dat的header
            temp = os.getcwd()
            settings_path = os.path.join(temp, 'modules/dingshan/settings.yaml')
            with open(settings_path, 'r', encoding='utf8') as f:
                s_y = yaml.load(f.read(), Loader=yaml.FullLoader)
                topic_list = s_y['Pasers'][0]['TopicNames']  # DatStatus 文件夹路径

            stream_dict = dict()
            stream_info_dict = dict()

            for topic in topic_list:
                stream_dict[topic] = []
                stream_info_dict[topic] = []

            # 取出topic内容
            while True:
                file, topic_name, topic_size, topic_timestamp = read_topic(file)
                if topic_name.strip('\x00') in topic_list:
                    topic_content = file.read(topic_size[0])

                    ## 暂存raw数据
                    try:
                        os.mkdir(f"{output_path}/{topic_name}")
                    except:
                        pass
                    # binfile = open(f"./output/{topic_name}/{topic_timestamp}.bin", "wb+")
                    # binfile.write(topic_content)
                    # binfile.close()

                    ## 解proto消息
                    message = encode_srv_pb2.EncodeH265()
                    message.ParseFromString(topic_content)
                    cam_id = message.camid
                    frame_id = message.frame_id
                    frame_size = message.frame_size
                    frame_data = message.frame_data
                    encodetype = message.encodetype
                    encode_width = message.encode_width
                    encode_height = message.encodetype
                    frame_ratenum = message.frame_ratenum
                    timestamp = message.timestamp
                    #
                    # print(
                    #     f'cam_id: {cam_id}, frame_id： {frame_id}, encodetype: {encodetype}, frame_ratenum: {frame_ratenum}, timestamp: {timestamp}')

                    ## 找到NAL单元
                    start_code = b"\x00\x00\x00\x01"
                    index = frame_data.find(start_code)
                    if index != -1:
                        frame_data = frame_data[index:]

                    start_code_vps = b"\x00\x00\x00\x01\x40\x01"
                    start_code_sps = b"\x00\x00\x00\x01\x42\x01"
                    start_code_pps = b"\x00\x00\x00\x01\x44\x01"
                    start_code_idr = b"\x00\x00\x00\x01\x26\x01"
                    start_code_slice = b"\x00\x00\x00\x01\x02\x01"

                    vps_index = frame_data.find(start_code_vps)
                    # if vps_index != -1:
                    #     print("vps")
                    sps_index = frame_data.find(start_code_sps)
                    # if sps_index != -1:
                    #     print("sps")
                    pps_index = frame_data.find(start_code_pps)
                    # if pps_index != -1:
                    #     print("pps")
                    idr_index = frame_data.find(start_code_idr)
                    # if idr_index != -1:
                    #     print("idr")
                    slice_index = frame_data.find(start_code_slice)
                    # if slice_index != -1:
                    #     print("slice")

                    ## 过滤无法解析的B slice
                    if len(stream_dict[topic_name.strip('\x00')]) == 0:
                        if vps_index != -1:
                            stream_dict[topic_name.strip('\x00')] = frame_data
                            stream_info_dict[topic_name.strip('\x00')] = [timestamp]
                    else:
                        stream_dict[topic_name.strip('\x00')] += frame_data
                        stream_info_dict[topic_name.strip('\x00')].append(timestamp)

                else:
                    file.seek(topic_size[0], 1)  # 跳过这一帧的topic长度
                if file.tell() >= file_size:  # 超过文件大小，退出
                    break
            
            # print('=================================>get image by whole bit stream')
            for topic in topic_list:
                if len(stream_dict[topic]) != 0:
                    print(f'convert topic: {topic}')
                    get_image2(stream_dict[topic], topic, stream_info_dict[topic], output_path)
                    # binfile = open(f"{output_path}/{topic}_stream.bin", "wb+")
                    # binfile.write(stream_dict[topic])
                    # binfile.close()
            


if __name__ == '__main__':
    output_path = "./img_out"  # 图像输出路径（按照摄像头名称区分）
    
    # dat_file = "/data/gyx/cqc_p2_cd701_raw/datasets_A5_0327/dats/CD701_000052__2024-03-27_11-41-15.dat"

    # 从dat_files.json中读取.dat文件路径
    with open('dat_files.json', 'r') as f:
        dat_files = json.load(f)
        # if dat_files:
        #     dat_file = dat_files[0]  # 使用第一个路径作为默认路径
        dat_file_list = dat_files  # 将所有路径读取到一个列表中


    for dat_file in dat_file_list:
        dat_path = Path(output_path) / Path(dat_file).stem
        if dat_path.exists():
            shutil.rmtree(dat_path)
        os.makedirs(dat_path)

        parser_topic(output_path, dat_file)
