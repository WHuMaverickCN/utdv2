import argparse
import os
from pathlib import Path

# 这几个引用不能删除
from modules.parser.S7.RTKRtcm_S7 import RTKRtcm_S7
from modules.parser.S7.GnssRtcm_S7 import GnssRtcm_S7
# from modules.parser.S7.Imu_S7 import Imu_S7s
from modules.parser.cd701.Ins_701 import Ins_701
# from modules.parser.cd701.Video_701 import Video_701
# 这几个引用不能删除
from modules.parser.config.settings import Settings
from modules.split.ReadDat import ReadDat
from modules.split.SplitFile import SplitFile

s_yaml = Settings.setting_yaml
sp = SplitFile(s_yaml)

from potime import RunTime

parsers_dict = {}

def get_parsers():
    if len(parsers_dict) == 0:
        for p in s_yaml[Settings.parser_key_name]:
            """
            反射创建对象：
            https://blog.csdn.net/weixin_48183870/article/details/135431744
            https://shanml.blog.csdn.net/article/details/79107711
            """
            global_class = globals()
            obj = global_class[p[Settings.parser_name]](s_yaml[Settings.parser_key_name][0][Settings.output_key_name][0])
            parsers_dict[p[Settings.parser_name]] = obj  # parser只创建一次
        return parsers_dict
    else:
        return parsers_dict


parsers_dict = get_parsers()


def process_dat_list(dat_list):
    push_status = False
    for file in dat_list:
        dat_path = Path(s_yaml[Settings.parser_key_name][0][Settings.output_key_name][0]) / Path(file).stem
        # dat_path = Path(s_yaml[Settings.output_key_name]) / Path(file).stem
        for parser_group in s_yaml[Settings.parser_key_name]:
            # parser_obj = Idmap_701(s_yaml[Settings.output_key_name])
            parser_obj = parsers_dict[parser_group[Settings.parser_name]]  # 拿到当前topic对应的parser的对象
            parser_obj.set_dat_name(Path(file).stem)
            r_d = ReadDat(dat_path, parser_obj)  # 创建读取文件的工具类
            for topic_name in parser_group[Settings.parser_topic_list]:  # 遍历topic
                push_status = r_d.push_each_frame(topic_name)  # 解析当前topic的所有帧
                if push_status == False:
                    return push_status
                # r_d.get_dat_frame_list(t, dat_index_list)  # 读取每一帧数据
            push_status = parser_obj.parser_data()  # 写出当前topic的处理结果
            if push_status == False:
                return push_status
    return push_status


# @RunTime
def process_dat(dat_file):
    dat_split_path = sp.split_dat(dat_file)
    if not dat_split_path or not dat_file.endswith('.dat'):
        return
    process_dat_list(dat_split_path)


def process_dir(dir_path):
    files = sp.split_dir(dir_path)
    process_dat_list(files)


if __name__ == '__main__':
    import multiprocessing

    # 创建一个ArgumentParser对象
    parser = argparse.ArgumentParser()

    # 添加一个名为'--task'的参数，帮助信息为'task json的位置'
    parser.add_argument('--file', help='1个dat的位置', default=None)
    parser.add_argument('--task', help='1个task的位置', default=None)
    parser.add_argument('--dir', help='1个dir的位置', default=None)
    parser.add_argument('--output', help='输出文件夹的位置', default=None)

    # 解析命令行参数
    args = parser.parse_args()

    dat_file = args.file
    task_file = args.task
    dir_path = args.dir

    args.output = './rsm'
    # dat_file = r'F:\000052车\0325_route2-8\CD701_000052__2024-03-25_14-03-25.dat'
    # process_dat(dat_file)
    # dir_path = r'C:\Users\202207817\Desktop\dat'  # dat文件所在目录
    dir_path = '/data/gyx/cqc_p2_cd701_raw/datasets_A5_0327/dats/'
    for file in os.listdir(dir_path):
        if not file.endswith('.dat'):
            continue
        file_path = os.path.join(dir_path, file)
        file_name = file.split('.')[0]
        process_dat(file_path)

    # pool.close()
    # pool.join()
    # task_file = r'./task_test.json'
    # if dat_file != None:
    #     process_dat(dat_file)
    # dir_path = r'C:\work\dat\all\4'
    # if dir_path != None:
    #     process_dir(dir_path)
    # if task_file != None:
    #     sp.split_task(task_file)
