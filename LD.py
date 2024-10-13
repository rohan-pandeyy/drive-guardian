import numpy as np
import cv2
import pickle
import torch
from tracker import tracker
import pandas as pd
import paho.mqtt.client as mqtt

# Set up the MQTT client
broker_address = "192.168.29.216"  # Replace with your broker address
mqtt_client = mqtt.Client(client_id="Publisher")
mqtt_client.connect(broker_address, 1883, 60)

# Load calibration data
dist_pickle = pickle.load(open("calibration_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


def yolo_detect(frame):
    # Perform YOLOv5 object detection
    results = model(frame)

    # Convert results to a DataFrame if not already (Ensure this is needed based on your model output)
    if not isinstance(results, pd.DataFrame):
        results_df = results.pandas().xyxy[0]
    else:
        results_df = results

    # Filter detections based on a higher confidence threshold
    if not results_df.empty:
        high_confidence_results = results_df[results_df['confidence'] > 0.5]
    
    return frame

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
    gradmag = np.sqrt(sobelx*2 + sobely*2)
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

def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height), max(0, int(center-width/2)):min(int(center+width/2), img_ref.shape[1])] = 1
    return output

def process_image(image):
    # Undistort the image
    img = cv2.undistort(image, mtx, dist, None, mtx)

    # Apply gradient and color thresholding
    gradx = abs_sobel_thresh(img, orient='x', thresh=(12, 255))
    grady = abs_sobel_thresh(img, orient='y', thresh=(25, 255))
    c_binary = color_threshold(img, sthresh=(100, 255), vthresh=(50, 255))
    preprocessImage = np.zeros_like(img[:, :, 0])
    preprocessImage[((gradx == 1) & (grady == 1)) | (c_binary == 1)] = 255

    # Perspective transform to get a bird's eye view
    img_size = (img.shape[1], img.shape[0])
    bot_width = 0.76  # Bottom trapezoid width
    mid_width = 0.08  # Top trapezoid width
    height_pct = 0.62  # Trapezoid height
    bottom_trim = 0.935  # Trim from the bottom of the image
    src = np.float32([[img.shape[1] * (0.5 - mid_width / 2), img.shape[0] * height_pct],
                      [img.shape[1] * (0.5 + mid_width / 2), img.shape[0] * height_pct],
                      [img.shape[1] * (0.5 + bot_width / 2), img.shape[0] * bottom_trim],
                      [img.shape[1] * (0.5 - bot_width / 2), img.shape[0] * bottom_trim]])
    offset = img_size[0] * 0.25
    dst = np.float32([[offset, 0], [img_size[0] - offset, 0],
                      [img_size[0] - offset, img_size[1]],
                      [offset, img_size[1]]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(preprocessImage, M, img_size, flags=cv2.INTER_LINEAR)

    # Lane detection using a sliding window
    window_width = 25
    window_height = 80
    curve_centers = tracker(Mywindow_width=window_width, Mywindow_height=window_height,
                            Mymargin=25, My_ym=10 / 720, My_xm=4 / 384, Mysmooth_factor=15)
    window_centroids = curve_centers.find_window_centroids(warped)

    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)
    rightx = []
    leftx = []

    for level in range(0, len(window_centroids)):
        l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
        r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)
        leftx.append(window_centroids[level][0])
        rightx.append(window_centroids[level][1])
        l_points[(l_points == 255) | ((l_mask == 1))] = 255
        r_points[(r_points == 255) | ((r_mask == 1))] = 255

    template = np.array(r_points + l_points, np.uint8)
    zero_channel = np.zeros_like(template)
    template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)
    warpage = np.array(cv2.merge((warped, warped, warped)), np.uint8)
    result = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)

    yvals = range(0, warped.shape[0])
    res_yvals = np.arange(warped.shape[0] - (window_height / 2), 0, -window_height)
    left_fit = np.polyfit(res_yvals, leftx, 2)
    left_fitx = left_fit[0] * yvals * yvals + left_fit[1] * yvals + left_fit[2]
    left_fitx = np.array(left_fitx, np.int32)
    right_fit = np.polyfit(res_yvals, rightx, 2)
    right_fitx = right_fit[0] * yvals * yvals + right_fit[1] * yvals + right_fit[2]
    right_fitx = np.array(right_fitx, np.int32)

    left_lane = np.array(list(zip(np.concatenate((left_fitx - window_width / 2, left_fitx[::-1] + window_width / 2), axis=0),
                                  np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
    right_lane = np.array(list(zip(np.concatenate((right_fitx - window_width / 2, right_fitx[::-1] + window_width / 2), axis=0),
                                   np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
    inner_lane = np.array(list(zip(np.concatenate((left_fitx + window_width / 2, right_fitx[::-1] - window_width / 2), axis=0),
                                   np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)

    lane_image = np.zeros_like(img)
    cv2.fillPoly(lane_image, [left_lane], [255, 0, 0])
    cv2.fillPoly(lane_image, [right_lane], [0, 0, 255])
    cv2.fillPoly(lane_image, [inner_lane], [0, 255, 0])

    # Unwarp the lane area back to original perspective
    lane_image_unwarped = cv2.warpPerspective(lane_image, Minv, (img.shape[1], img.shape[0]))
    result = cv2.addWeighted(img, 1, lane_image_unwarped, 0.3, 0)

    # YOLO object detection
    results = model(img)
    detections = results.pandas().xyxy[0]

    # Create a binary mask of the lane area
    lane_mask = cv2.cvtColor(lane_image_unwarped, cv2.COLOR_BGR2GRAY)
    _, lane_mask = cv2.threshold(lane_mask, 1, 255, cv2.THRESH_BINARY)

    brake_signal = 1  # Initialize as not braking (0)

    # Draw detections on the image
    for _, detection in detections.iterrows():
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        conf, cls = detection['confidence'], detection['name']

        box_area = lane_mask[y1:y2, x1:x2]

        box_color = (0, 255, 0)  # Default green

        if np.any(box_area):
            bottom_center_y = y2
            distance_ratio = (bottom_center_y) / image.shape[0]

            if distance_ratio > 0.8:
                box_color = (0, 0, 255)  # Red
                brake_signal = 1  # Brake (1)
            elif distance_ratio > 0.5:
                box_color = (0, 255, 255)  # Yellow

        label = f"{cls} {conf:.2f}"
        cv2.rectangle(result, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(result, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # Publish the brake signal over MQTT
    mqtt_client.publish("test", str(brake_signal))
    
    return result

def process_video_realtime(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # First apply YOLO detections on the frame
        frame_with_detections = yolo_detect(frame)

        # Then, process the frame for lane detection
        processed_frame = process_image(frame_with_detections)
        
        # Display the processed frame
        cv2.imshow('Lane Detection and Object Detection', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()