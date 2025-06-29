# 用于原始数据包处理的类
class RawDataUtils:
    @staticmethod
    def get_raw_data_path():
        """
        获取原始数据包的路径
        :return: 原始数据包的路径
        """
        return "./data/raw_data/"

    @staticmethod
    def get_processed_data_path():
        """
        获取处理后的数据包的路径
        :return: 处理后的数据包的路径
        """
        return "./data/processed_data/"