import numpy as np
import cv2
import pickle
import torch
from tracker import tracker
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression
from utils.torch_utils import select_device

# Load camera calibration data
dist_pickle = pickle.load(open("calibration_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Initialize YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    with np.errstate(divide='ignore', invalid='ignore'):
        absgraddir = np.absolute(np.arctan(sobely/sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output

def color_threshold(image, sthresh=(0, 255), vthresh=(0, 255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])] = 1
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= vthresh[0]) & (v_channel <= vthresh[1])] = 1
    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary == 1)] = 1
    return output

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output


def yolo_detect(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)
    return results.xyxy[0].cpu().numpy()  # Extract detection results

def process_image(frame, detections):
    img = cv2.undistort(frame, mtx, dist, None, mtx)
    preprocessImage = np.zeros_like(img[:,:,0])
    gradx = abs_sobel_thresh(img, orient='x', thresh=(12,255))
    grady = abs_sobel_thresh(img, orient='y', thresh=(25,255))
    c_binary = color_threshold(img, sthresh=(100,255), vthresh=(50,255))
    preprocessImage[((gradx == 1) & (grady == 1)) | (c_binary == 1)] = 255

    img_size = (img.shape[1], img.shape[0])
    src = np.float32([
        [img.shape[1]*(0.5-0.08/2), img.shape[0]*0.62],
        [img.shape[1]*(0.5+0.08/2), img.shape[0]*0.62], 
        [img.shape[1]*(0.5+0.76/2), img.shape[0]*0.935], 
        [img.shape[1]*(0.5-0.76/2), img.shape[0]*0.935]
    ])
    dst = np.float32([
        [img_size[0]*0.25, 0], 
        [img_size[0]*0.75, 0], 
        [img_size[0]*0.75, img_size[1]], 
        [img_size[0]*0.25, img_size[1]]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(preprocessImage, M, img_size, flags=cv2.INTER_LINEAR)

    curve_centers = tracker(Mywindow_width=25, Mywindow_height=80, Mymargin=25, My_ym=10/720, My_xm=4/384, Mysmooth_factor=15)
    window_centroids = curve_centers.find_window_centroids(warped)

    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)
    for level in range(0,len(window_centroids)):
        l_mask = window_mask(25, 80, warped, window_centroids[level][0], level)
        r_mask = window_mask(25, 80, warped, window_centroids[level][1], level)
        l_points[(l_points == 255) | (l_mask == 1)] = 255
        r_points[(r_points == 255) | (r_mask == 1)] = 255

    # Draw the bounding boxes for each detection
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        if conf > 0.5:  # Filter out low confidence detections
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Combine lane markings with the original image
    lane_warped = np.dstack((l_points, np.zeros_like(warped), r_points)) * 255
    lane_unwarped = cv2.warpPerspective(lane_warped, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    result = cv2.addWeighted(img, 1, lane_unwarped, 0.3, 0)

    return result

def process_video_realtime(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = yolo_detect(frame)
        processed_frame = process_image(frame, detections)
        cv2.imshow('Lane Detection and Object Detection', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# You will need to fill in the details for your `tracker` class and any helper functions used, like `abs_sobel_thresh`, `color_threshold`, etc., to ensure the functionality of the lane detection part.

