from pathlib import Path
import os
import cv2
import ffmpeg
import imageio
import av
import av.codec
import av.video
import PIL
"""
https://pyav.org/docs/develop/cookbook/basics.html
"""

def get_image2(img_bin, topic_name,timestamps, output_path):
    """
    目前只有软件解析，没有GPU解析
    """
    ## TODO 1、使用GPU解码

    rc = 0
    frames = []
    context = av.codec.context.CodecContext.create("hevc", "r")

    try:
        packets = context.parse(img_bin)
        # print("=====================>packets count: %d" % len(packets))
        for packet in packets:
            packet_frames = context.decode(packet)
            # print("=====================>packets frames count: %d" % len(packet_frames))
            frames_size = len(packet_frames)
            for i in range(frames_size):
                frames.append(packet_frames[i])
    except Exception as e:
        print("decode 异常")
        print(e)

    if len(timestamps) != len(frames):
        # print(f"len(timestamps): {len(timestamps)}, len(frames): {len(frames)}")
        pass ## 最后的P slice由于没有下一帧idr无法解析, 可以接受

    # print("=====================>frame count: %d" % len(frames))
    frame_index = 0
    for frame in frames:
        # print('Frame index：%d' % frame.index)
        # print('Key Frame：%d' % frame.key_frame)
        # print('Picture type：%s' % frame.pict_type)

        # print(f"try save image: ./output/{topic_name}/{timestamps[frame_index]}.jpg")
        try:
            img = frame.to_image()
            img.save(f"{output_path}/{topic_name}/{timestamps[frame_index]}.jpg", quality=100, )
            frame_index += 1
            # print("解析完成")
        except Exception as e:
            print("解析失败")
            continue
    rc = 1


    # codec.close()
    return rc
