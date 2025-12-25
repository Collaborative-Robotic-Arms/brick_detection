import math

class BrickTracker:
    def __init__(self, distance_threshold=60, max_disappeared=10):
        # Store state: {id: {'center': (x,y), 'type': 'L', 'disappeared': 0, 'smooth_vec': (vx, vy)}}
        self.tracked_objects = {}
        self.next_object_id = 0
        self.distance_threshold = distance_threshold
        self.max_disappeared = max_disappeared
        
        # Smoothing factor (0.0 = no new data, 1.0 = no smoothing). 
        # 0.2 means we trust the history 80% and new data 20%.
        self.alpha = 0.2 

    def update(self, detections):
        """
        detections: list of dicts.
        Returns: The list with 'id' and 'stable_angle' added.
        """
        # 1. Mark existing as potentially missing
        for obj_id in self.tracked_objects:
            self.tracked_objects[obj_id]['disappeared'] += 1

        # 2. Match Detections to IDs
        if len(detections) > 0:
            for det in detections:
                cx, cy = det['center']
                best_id = -1
                min_dist = self.distance_threshold

                for obj_id, data in self.tracked_objects.items():
                    # Strict type matching
                    if data['type'] != det['type']:
                        continue

                    dist = math.hypot(cx - data['center'][0], cy - data['center'][1])
                    if dist < min_dist:
                        min_dist = dist
                        best_id = obj_id

                # 3. Update State
                if best_id != -1:
                    # Found match
                    self.tracked_objects[best_id]['center'] = (cx, cy)
                    self.tracked_objects[best_id]['disappeared'] = 0
                    
                    # --- STABILIZE ANGLE ---
                    raw_angle = det['angle']
                    stable_angle = self.smooth_angle(best_id, raw_angle)
                    
                    det['id'] = best_id
                    det['angle'] = stable_angle # Overwrite with smooth angle
                else:
                    # New object
                    new_id = self.register(det)
                    det['id'] = new_id
                    # Initial angle is just the raw angle
                    det['angle'] = det['angle'] 

        # 4. Cleanup
        self.cleanup()
        return detections

    def smooth_angle(self, obj_id, raw_angle):
        """
        Smooths angle using vector averaging to handle the 360-wrap 
        and PCA 180-flip ambiguity.
        """
        # Get previous smoothed vector (vx, vy)
        prev_vx, prev_vy = self.tracked_objects[obj_id]['smooth_vec']
        
        # Convert new raw angle to vector
        new_vx = math.cos(raw_angle)
        new_vy = math.sin(raw_angle)

        # --- Fix 180-degree Flip (PCA Ambiguity) ---
        # If dot product is negative, the vectors point in opposite directions.
        # Flip the new one to match the "flow" of the previous one.
        dot_product = (prev_vx * new_vx) + (prev_vy * new_vy)
        if dot_product < 0:
            new_vx = -new_vx
            new_vy = -new_vy

        # --- Exponential Moving Average (EMA) ---
        # avg = (1-alpha)*old + alpha*new
        avg_vx = (1 - self.alpha) * prev_vx + self.alpha * new_vx
        avg_vy = (1 - self.alpha) * prev_vy + self.alpha * new_vy

        # Update stored vector
        self.tracked_objects[obj_id]['smooth_vec'] = (avg_vx, avg_vy)
        
        # Convert back to radians
        return math.atan2(avg_vy, avg_vx)

    def register(self, det):
        obj_id = self.next_object_id
        # Initialize smooth vector from the very first detection
        vx = math.cos(det['angle'])
        vy = math.sin(det['angle'])
        
        self.tracked_objects[obj_id] = {
            'center': det['center'],
            'type': det['type'],
            'disappeared': 0,
            'smooth_vec': (vx, vy) 
        }
        self.next_object_id += 1
        return obj_id

    def cleanup(self):
        ids_to_delete = [
            obj_id for obj_id, data in self.tracked_objects.items()
            if data['disappeared'] > self.max_disappeared
        ]
        for obj_id in ids_to_delete:
            del self.tracked_objects[obj_id]
