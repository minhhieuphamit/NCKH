import numpy as np
from ultralytics import YOLO
import cv2
from numpy import random
import torch

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque

model = YOLO('models/yolov8l.pt')
deepsort = None
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}

object_counter = {}
cap = cv2.VideoCapture('videos/demo.mp4')

my_file = open("names/coco2.txt", "r")
data = my_file.read()
names = data.split("\n")

# line2 = [(362,496), (1253,191)]
# line2 = [(361, 559),(772, 571)]
# line2 = [272, 636],[1212, 632]
# THuThiem1
# line2 =[960, 941],[1384, 953]

# BinhLoi2
line2 = [465, 878],[1577, 842]
# line2 = [(294,413), (785,555)]
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

##Thu Thiem1
# pts = np.array([[936, 968],[1392, 980],[1292, 608],[1136, 600],[936, 968]])
##Binh loi 2
pts = np.array([[396, 968], [1692, 948], [1092, 352], [824, 368], [396, 968]])


# pts =np.array([[348, 592],[472, 332],[540, 184],[560, 140],[696, 140],[724, 280],[756, 448],[780, 600],[344, 592]])
# pts = np.array([[276, 647],[272, 411],[368, 315],[668, 311],[716, 343],[772, 363],[824, 383],[892, 427],[1216, 647],[272, 647]])
# pts =np.array([[144, 335],[248, 267],[296, 231],[316, 215],[340, 203],[376, 187],[404, 171],[416, 155],[520, 151],[548, 195],[608, 287],[676, 391],[712, 443],[4, 431],[144, 335]])
def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 2:  # Xe buyt
        color = (0, 149, 255)
    elif label == 3:  # xe hoi
        color = (222, 82, 175)
    elif label == 5:  # Xe may
        color = (0, 204, 255)
    elif label == 7:  # xe tai
        color = (85, 45, 255)
    else:
        color = (255, 255, 255)
    return tuple(color)


######

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)

    cv2.circle(img, (x1 + r, y1 + r), 2, color, 12)
    cv2.circle(img, (x2 - r, y1 + r), 2, color, 12)
    cv2.circle(img, (x1 + r, y2 - r), 2, color, 12)
    cv2.circle(img, (x2 - r, y2 - r), 2, color, 12)

    return img


def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        # c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3

        img = draw_border(img, (c1[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0], c1[1] + 3), color, 1, 8, 2)

        # cv2.line(img, c1, c2, color, 30)
        # cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def draw_boxes(img, bbox, object_id, identities=None):
    area_roi = cv2.contourArea(pts)
    mylist = []
    xemay = []
    xehoi = []
    xebuyt = []
    xetai = []
    height, width, _ = img.shape


    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        center = (int((x2 + x1) / 2), int((y2 + y2) / 2))

        dist = cv2.pointPolygonTest(pts, (center), False)

        if (dist >= 0):
            #UI_box(box, img, label=label, color=color, line_thickness=2)
            id = int(identities[i]) if identities is not None else 0

            # create new buffer for new object
            if id not in data_deque:
                data_deque[id] = deque(maxlen=64)
            color = compute_color_for_labels(object_id[i])
            obj_name = names[object_id[i]]
            label = '%s' % (obj_name)

            # add center to buffer
            data_deque[id].appendleft(center)
            if len(data_deque[id]) >= 2 :
                if intersect(data_deque[id][0], data_deque[id][1], line2[0], line2[1]):
                    cv2.line(img, line2[0], line2[1], (255, 255, 255), 3)
                    if obj_name not in object_counter:
                        object_counter[obj_name] = 1
                    else:
                        object_counter[obj_name] += 1
            # draw trail
            for i in range(1, len(data_deque[id])):
                # check if on buffer value is none
                if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                    continue
                # generate dynamic thickness of trails
                thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
                # draw trails
                cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)
            width_b = abs(box[2] - box[0])
            height_b = abs([box[3]] - box[1])
            area = width_b * height_b
            UI_box(box, img, label=label, color=color, line_thickness=2)
            if (label == "xe may"):
                xemay.append(area)
            if (label == "xe hoi"):
                xehoi.append(area)
            if (label == "xe tai"):
                xetai.append(area)
            if (label == "xe buyt"):
                xebuyt.append(area)


    count = 0
    for idx, (key, value) in enumerate(object_counter.items()):
        cnt_str1 = str(key) + ":" + str(value)
        print(str(key))
        cv2.line(img, (20, 25), (300, 25), [85, 45, 255], 40)
        cv2.putText(img, f'Dem so luong xe : ', (11, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        cv2.line(img, (20, 65 + (idx * 40)), (180, 65 + (idx * 40)), [85, 45, 255], 30)
        cv2.putText(img, cnt_str1, (11, 75 + (idx * 40)), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        count += value
        cv2.line(img, (20, 65 + ((idx + 1) * 40)), (150, 65 + ((idx + 1) * 40)), [91, 189, 43], 30)
        cv2.putText(img, "Tong :" + str(count), (11, 75 + ((idx + 1) * 40)), 0, 1, [225, 255, 255], thickness=2,
                    lineType=cv2.LINE_AA)
    agv_area_xemay = np.nan_to_num(np.mean(xemay, dtype=np.float64))
    sum_area_xemay = agv_area_xemay * len(xemay)

    # print(str(density_xemay_rou))

    # ---------------------------------

    agv_area_xehoi = np.nan_to_num(np.mean(xehoi, dtype=np.float64))
    sum_area_xehoi = agv_area_xehoi * len(xehoi)

    # print(str(density_xehoi_rou))

    # ---------------------------------------
    agv_area_xetai = np.nan_to_num(np.mean(xetai, dtype=np.float64))
    sum_area_xetai = agv_area_xetai * len(xetai)

    # ---
    agv_area_xebuyt = np.nan_to_num(np.mean(xebuyt, dtype=np.float64))
    sum_area_xebuyt = agv_area_xebuyt * len(xebuyt)

    density_total = round(((sum_area_xebuyt + sum_area_xetai + sum_area_xemay + sum_area_xehoi) / area_roi), 2)

    return img, count, density_total


####################################################
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))
count_frame = 0
dem = 0
# initialize deepsort
cfg_deep = get_config()
cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
# attempt_download("deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7", repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                    max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                    max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT,
                    nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                    use_cuda=True)
density_total = 0
density = deque()
density_final = 0
while True:
    ret, frame = cap.read()
    cv2.line(frame, line2[0], line2[1], (255, 0, 112), 3)
    results = model.predict(frame, conf=0.45, iou=0.5, device=0, classes=[2, 3, 5, 7])
    a = results[0].boxes.cuda()
    boxes = a.cpu().numpy()
    xywh = boxes.xywh

    confs = boxes.conf
    oids = []

    for cls in boxes.cls:
        oid = int(cls)
        oids.append(oid)
    xywh_tensor = torch.Tensor(xywh)
    confs_tensor = torch.Tensor(confs)
    outputs = deepsort.update(xywh_tensor, confs_tensor, oids, frame)

    mask_ROI = np.zeros_like(frame)
    cv2.fillPoly(mask_ROI, [pts], (128, 255, 0))
    cv2.polylines(mask_ROI, [pts], True, (255, 255, 128), 2)
    frame = cv2.addWeighted(frame, 1, mask_ROI, 0.2, 0)

    if (len(outputs)) > 0:
        bbox_xyxy = outputs[:, :4]
        identities = outputs[:, -2]
        object_id = outputs[:, -1]
        offset = (0, 0)
        count_frame += 1

        for i, box in enumerate(bbox_xyxy):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            color = compute_color_for_labels(object_id[i])
            frame, count, density_total = draw_boxes(frame, bbox_xyxy, object_id, identities)
    density.append(density_total)
    if (count_frame % 150 == 0):
        density_avg = sum(density) / len(density)
        density_final = round((density_avg * 100), 2)
        density.clear()
    cv2.line(frame, (width - 300, 25), (width - 50, 25), [85, 45, 255], 40)
    cv2.putText(frame, "Mat do: " + str(density_final) + " %", (width - 300, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
    cv2.imshow('Results video', frame)
    # ghi khung hình đầu vào vào video đầu ra
    out.write(frame)
    # nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('frame.png', frame)
        break

# giải phóng tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()
