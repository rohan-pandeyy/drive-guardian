class LaneDepartureWarning:
    def __init__(self, drift_threshold: int = 50):
        self.drift_threshold = drift_threshold

    def evaluate(self, lane_results, frame_width, frame_height):
        """
        Evaluates UFLD lane lines to determine if the vehicle is drifting.
        Accepts:
            lane_results: List of lanes, where each lane is a list of (x,y) points.
            frame_width, frame_height: Ints representing the frame dimensions.
        Returns:
            is_drifting (bool), warning_message (str)
        """
        if not lane_results or len(lane_results) < 2:
            return False, ""
            
        # The center of the camera frame represents the "hood" of our car
        hood_x = frame_width // 2
        
        # Extract the bottom-most points of all detected lanes (highest Y value)
        bottom_points = []
        for lane in lane_results:
            if len(lane) > 0:
                # Sort lane points by Y coordinate descending (bottom of screen first)
                sorted_lane = sorted(lane, key=lambda p: p[1], reverse=True)
                bottom_points.append(sorted_lane[0])
                
        # Sort these bottom points by their X coordinates so we can find Left and Right
        bottom_points = sorted(bottom_points, key=lambda p: p[0])
        
        # Find the two lines that border the center hood
        left_line_x = None
        right_line_x = None
        
        for p in bottom_points:
            if p[0] < hood_x:
                left_line_x = p[0] # The closest one on the left
            elif p[0] > hood_x and right_line_x is None:
                right_line_x = p[0] # The closest one on the right
                
        is_drifting = False
        msg = ""

        # Check drift strictly against the closest defining ego-lanes
        if left_line_x is not None:
            if abs(hood_x - left_line_x) < self.drift_threshold:
                is_drifting = True
                msg = "WARNING: DRIFTING LEFT"
                
        if right_line_x is not None:
            if abs(right_line_x - hood_x) < self.drift_threshold:
                is_drifting = True
                msg = "WARNING: DRIFTING RIGHT"
                
        return is_drifting, msg
