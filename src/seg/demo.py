import argparse
import time
from pathlib import Path
import cv2
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import os 
import random
import shutil
print(os.getcwd())
# Conclude setting / general reprocessing / plots / metrices / datasets
from utils.utils import \
    time_synchronized,select_device, increment_path,\
    scale_coords,xyxy2xywh,non_max_suppression,split_for_trace_model,\
    driving_area_mask,lane_line_mask,plot_one_box,show_seg_result,extra_save_ll_mask_to_image,extra_save_ll_to_array,\
    AverageMeter,\
    LoadImages

# 定义一个装饰器，用于统计函数运行时间
def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 开始时间
        result = func()  # 执行函数
        end_time = time.time()  # 结束时间
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to complete.")
        return result
    return wrapper

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='data/weights/seg_model.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/example.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--jpg-mode', action='store_true', help='使用jpg格式存储分割数据,此方法节约存储空间与传输带宽')
    parser.add_argument('--sampling-rate', type=float, default=0.2, help='proportion of images to process (0.0-1.0)')
    return parser


@timeit
def detect():
    # Get the current project directory
    current_dir = os.getcwd()
    # Check if 'seg' is in the current directory path
    if 'seg' not in current_dir:
        # Adjust the base directory by adding 'src/seg'
        current_dir = os.path.join(current_dir, 'src/seg')
        # Update the working directory
        os.chdir(current_dir)
    print(f"Current project directory: {current_dir}")

    # "额外处理"
    opt.weights = os.path.join(os.getcwd(),opt.weights)
    # Update project path to the parent directory of the source path
    source_path = Path(opt.source)
    if source_path.is_file():
        # If source is a file, remove file name and its containing directory
        opt.project = str(source_path.parent.parent)
    else:
        # If source is a directory, remove just the last directory
        opt.project = str(source_path.parent)

    # setting and directories
    source, weights,  save_txt, imgsz = opt.source, opt.weights,  opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    # Extract the last directory from the source path
    source_path = Path(opt.source)
    # Get the last directory name
    last_dir = source_path.parent.name if source_path.is_file() else source_path.name
    last_dir = last_dir if last_dir else 'default'  # Use 'default' if empty

    # Create save directory with last_dir + '_seg' suffix
    # save_dir = Path(increment_path(Path(opt.project) / (last_dir + '_seg'), exist_ok=opt.exist_ok))
    save_dir = Path(opt.project) / (last_dir + '_seg')
    # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # Check if save_dir exists and clear it if it does
    if save_dir.exists():
        print(f"Clearing existing directory: {save_dir}")
        # Remove all files and subdirectories
        for item in save_dir.glob('*'):
            if item.is_file():
                item.unlink()  # Delete file
            elif item.is_dir():
                shutil.rmtree(item)  # Delete directory and all its contents
        print(f"Directory cleared: {save_dir}")
    # Create directory structure
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()
    # 开始记录时间
    start_time = time.time()

    # Load model
    stride = 32
    device = select_device(opt.device)
    model = torch.jit.load(weights, map_location=torch.device('cpu') if not device.type!='cpu' else None)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = model.to(device)

    if half:
        model.half()  # to FP16  
    model.eval()

    end_time = time.time()
    print(f"Model loading and preparation took {end_time - start_time:.4f} seconds to complete.")
    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    # Apply sampling rate if less than 1.0
    if opt.sampling_rate < 1.0:
        print(f"Sampling at rate {opt.sampling_rate:.2f}")
        
        # Override the dataset's iteration behavior to only yield sampled items
        # Create dataset for initial file list
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        
        # Get the number of items to sample
        total_items = len(dataset)
        sample_count = max(1, int(total_items * opt.sampling_rate))
        
        # Create indices for evenly spaced sampling
        interval = total_items / sample_count
        sampled_indices = sorted([int(interval * i) for i in range(sample_count)])
        
        # Ensure the last index doesn't exceed the dataset size
        if sampled_indices and sampled_indices[-1] >= total_items:
            sampled_indices[-1] = total_items - 1
        
        # Create a new dataset with only the sampled files
        sampled_files = [dataset.files[i] for i in sampled_indices]
        sampled_video_flag = [dataset.video_flag[i] for i in sampled_indices]
        
        # Replace the dataset's files with the sampled ones
        dataset.files = sampled_files
        dataset.video_flag = sampled_video_flag
        dataset.nf = len(sampled_files)
        print(f"Sampled {sample_count} items from {total_items} total items")
    else:
        # Create dataset without sampling
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        [pred,anchor_grid],seg,ll= model(img)
        t2 = time_synchronized()

        # waste time: the incompatibility of  torch.jit.trace causes extra time consumption in demo version 
        # but this problem will not appear in offical version 
        tw1 = time_synchronized()
        pred = split_for_trace_model(pred,anchor_grid)
        tw2 = time_synchronized()

        # Apply NMS
        t3 = time_synchronized()
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t4 = time_synchronized()

        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)
        # 此处ll_seg_mask为车道线分割图，da_seg_mask为车道区域分割图
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # _pkl_uuid = p.parts[7]
            _pkl_mask_name = p.name.replace(".jpg", "_seg_mask.pkl")

            # _img_mask_name = p.name.replace(".jpg", "_seg_mask.jpg")
            # pkl_mask_save_path = str(save_dir / _pkl_mask_name)
            pkl_mask_save_path = str(save_dir /_pkl_mask_name)
            # pkl_mask_save_path = str(save_dir /p.parts[-3] / "array_mask"/_pkl_mask_name)
            save_path = str(save_dir / p.name)  # img.jpg
            # image_mask_save_path = str(save_dir /p.parts[7] / "image_mask"/_img_mask_name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img :  # Add bbox to image
                        plot_one_box(xyxy, im0, line_thickness=3)

            # Print time (inference)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            show_seg_result(im0, (da_seg_mask,ll_seg_mask), is_demo=True)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    # cv2.imwrite(save_path, im0)
                    # extra_save_ll_mask_to_image(ll_seg_mask,save_path)
                    extra_save_ll_to_array(ll_seg_mask,pkl_mask_save_path)
                    # extra_save_ll_mask_to_image(ll_seg_mask,image_mask_save_path)
                    print(f" The image with the result is saved in: {pkl_mask_save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            #w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            #h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            w,h = im0.shape[1], im0.shape[0]
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    inf_time.update(t2-t1,img.size(0))
    nms_time.update(t4-t3,img.size(0))
    waste_time.update(tw2-tw1,img.size(0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))
    print(f'Done. ({time.time() - t0:.3f}s)')

if __name__ == '__main__':
    opt =  make_parser().parse_args()
    print(opt)

    with torch.no_grad():
        detect(1000000)
