import json

from pofile import mkdir

from .config.settings import *


class BaseParser(object):
    def __init__(self, output_path, outdir=None):
        self.output_path = Path(output_path).absolute()
        if outdir:
            self.out_dir = outdir
        else:
            self.out_dir = ''
        self.out_dat_path = None
        self.topic_name = None

    def set_dat_name(self, set_dat_name):
        self.out_dat_path = self.output_path / set_dat_name / self.out_dir
        mkdir(self.out_dat_path)

    def set_topic_name(self, topic_name):
        self.topic_name = topic_name

    def write_json_file(self, file_name, file):
        """
        结果保存为json文件
        :param file_name:
        :param file:
        :return:
        """
        with open(file_name, 'w') as res_file:
            json.dump(file, res_file, indent=4)

    def write_img_file(self, file_name, file):
        """
        结果保存为图片
        :param file_name:
        :param file:
        :return:
        """
        pass

    def write_bin_file(self, file_name, data):
        """
        结果保存为二进制文件
        :param file_name:
        :param file:
        :return:
        """
        with open(file_name, 'ab') as res_file:
            res_file.write(data)

    def push_data(self, data,topic_name=None):
        """
        依次读取dat内容，并且解析后保存起来
        :return:
        """
        pass

    def parser_data(self):
        """
        解析一帧数据
        :return:
        """
        pass
