from pathlib import Path

import cv2
import ffmpeg
import av
import imageio

"""
https://pyav.org/docs/develop/cookbook/basics.html
"""


def get_image(img_bin, topic_name, timestamp):
    """
    Get image from binary data
    """
    # try:
    codec = av.CodecContext.create("hevc_cuvid", "r")
    p = av.Packet(img_bin)
    frames = codec.decode(p)
    for frame in frames:
        Path.mkdir(topic_name, exist_ok=True)
        frame.to_image().save(f"./{topic_name}/{timestamp[0]}.jpg", quality=100, )

    codec.close()
    # except:
    #     print("不是关键帧")
