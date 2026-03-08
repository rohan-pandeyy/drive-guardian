class ForwardCollisionWarning:
    def __init__(self, critical_area_threshold: float = 0.15):
        # If a car takes up more than 15% of the frame, it's critically close.
        self.critical_area_threshold = critical_area_threshold

    def evaluate(self, yolo_results, lane_results, frame_width, frame_height):
        """
        Evaluates YOLO bounding boxes against UFLD ego-lanes to determine collision risk.
        Accepts:
            yolo_results: YOLO inference object.
            lane_results: List of UFLD lane lines.
            frame_width, frame_height: Ints.
        Returns:
            is_crashing (bool), warning_message (str)
        """
        if yolo_results is None or len(yolo_results) == 0:
            return False, ""

        boxes = yolo_results.boxes
        if boxes is None or len(boxes) == 0:
            return False, ""

        # Establish Ego-Lane Boundaries (fallback to generic if lanes missing)
        left_lane_x = 0
        right_lane_x = frame_width
        
        # If we have lanes, narrow the ego-lane focus
        if lane_results and len(lane_results) >= 2:
            hood_x = frame_width // 2
            bottom_points = []
            for lane in lane_results:
                if len(lane) > 0:
                    sorted_lane = sorted(lane, key=lambda p: p[1], reverse=True)
                    bottom_points.append(sorted_lane[0])
                    
            bottom_points = sorted(bottom_points, key=lambda p: p[0])
            for p in bottom_points:
                if p[0] < hood_x:
                    left_lane_x = p[0]
                elif p[0] > hood_x and right_lane_x == frame_width:
                    right_lane_x = p[0]

        total_frame_area = frame_width * frame_height
        is_crashing = False
        
        # Loop through detected YOLO boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            # We only care about cars, trucks, buses (standard COCO IDs: 2, 5, 7)
            if cls_id not in [2, 5, 7]:
                continue
                
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            box_width = x2 - x1
            box_height = y2 - y1
            box_bottom_center_x = x1 + (box_width / 2)
            
            # 1. Is the car physically inside our ego-lane?
            if left_lane_x <= box_bottom_center_x <= right_lane_x:
                
                # 2. Is the car critically close?
                box_area = box_width * box_height
                if (box_area / total_frame_area) > self.critical_area_threshold:
                    is_crashing = True
                    break # We only need one critical threat to trigger the warning

        if is_crashing:
            return True, "WARNING: BRAKE!"
            
        return False, ""
