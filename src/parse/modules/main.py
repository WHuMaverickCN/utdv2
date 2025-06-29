import argparse
from pathlib import Path

# 这几个引用不能删除
from parser.S7.RTKRtcm_S7 import RTKRtcm_S7
from parser.S7.GnssRtcm_S7 import GnssRtcm_S7
from parser.S7.Imu_S7 import Imu_S7
from parser.cd701.Ins_701 import Ins_701
from parser.cd701.Video_701 import Video_701
# 这几个引用不能删除
from parser.config.settings import Settings
from split.ReadDat import ReadDat
from split.SplitFile import SplitFile

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
            obj = global_class[p[Settings.parser_name]](s_yaml[Settings.output_key_name])
            parsers_dict[p[Settings.parser_name]] = obj  # parser只创建一次
        return parsers_dict
    else:
        return parsers_dict


parsers_dict = get_parsers()


def process_dat_list(dat_list):
    for file in dat_list:
        dat_path = Path(s_yaml[Settings.output_key_name]) / Path(file).stem
        for p in s_yaml[Settings.parser_key_name]:
            parser_obj = parsers_dict[p[Settings.parser_name]]  # 拿到当前topic对应的parser的对象
            parser_obj.set_dat_name(Path(file).stem)

            r_d = ReadDat(dat_path, parser_obj)  # 创建读取文件的工具类
            for t in p[Settings.parser_topic_list]:  # 遍历topic
                dat_index_list = r_d.get_dat_index_list(t)
                r_d.get_dat_frame_list(t, dat_index_list)  # 读取每一帧数据
            parser_obj.parser_data()


@RunTime
def process_dat(dat_file):
    dat_split_path = sp.split_dat(dat_file)
    process_dat_list(dat_split_path)

def process_dir(dir_path):
    files = sp.split_dir(dir_path)
    process_dat_list(files)


if __name__ == '__main__':
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
    output_path = args.output

    output_path = r'./output'
    # dat_file = r'/mnt/c/work/dat/all/4/S7_003494_2024-01-20_19-25-00.dat'
    # dat_file = r'/mnt/c/work/dat/all/3/S7_007067_2024-01-17_21-42-00.dat'
    # dat_file = r'C:\work\dat\all\test\cd701\CD701_000052__2024-03-19_13-34-10.dat'
    # dat_file = r'C:\work\dat\all\3/S7_007067_2024-01-17_21-42-00.dat'
    # dat_file = r'C:\work\dat\all\4/S7_003494_2024-01-20_19-25-00.dat'
    # dat_file = r'/mnt/c/work/dat/all/test/cd701/CD701_000052__2024-03-19_13-34-10.dat'
    dat_file = "/data/gyx/cqc_p2_cd701_raw/datasets_A5_0327/dats/CD701_000052__2024-03-27_11-50-29.dat"
    task_file = r'./task_test.json'
    if dat_file != None:
        process_dat(dat_file)
    # dir_path = r'C:\work\dat\all\4'
    # if dir_path != None:
    #     process_dir(dir_path)
    # if task_file != None:
    #     sp.split_task(task_file)
