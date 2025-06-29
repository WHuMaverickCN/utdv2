from pathlib import Path

import yaml

config_path = Path(__file__).parent.absolute()


class Settings:
    setting_path = config_path / "settings.yaml"
    setting_yaml = None
    with open(setting_path, 'r', encoding='utf8') as f:
        setting_yaml = yaml.load(f.read(), Loader=yaml.FullLoader)
    parser_key_name = 'Pasers'
    parser_name = 'PaserName'
    parser_topic_list = 'TopicNames'
    topic_key_name = 'TopicNames'
    output_key_name = 'OutDic'


DAT_SUFFIX = '.dat'
DAT_INDEX = '-index'
