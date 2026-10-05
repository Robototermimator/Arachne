# Imports
import cv2
import time
from ultralytics import YOLO

# -------------------------
# DECLARATIONS
# -------------------------

previous_frame = None
position_history = []


direction = "STATIONARY"

speed = 0
smoothed_speed = 0

movement_status = "STATIONARY"
no_motion_frames = 0

# -------------------------
# TARGET SYSTEM
# -------------------------

target_locked = False
target_id = 1
target_lost_frames = 0

tracked_position = None
tracked_box = None

target_velocity_x = 0
target_velocity_y = 0

# MOVEMENT VECTOR
vector_scale = 8

# PREDICTED TARGET POSITION
prediction_marker_size = 6

maximum_target_jump = 120
position_smoothing = 0.7

# Target Scoring
target_width = 0
target_height = 0

# TARGET DISTANCE
target_distance = 0
target_distance_status = "UNKNOWN"

#Distance Trend
target_distance_previous = 0
target_distance_trend = "STABLE"

#TRACKING TIMING 
target_tracking_time = 0
target_tracking_start = None

#--------------------------
#PREDICTION
#--------------------------
tracking_prediction_frames = 0
maximum_prediction_frames = 15
prediction_velocity_decay = 0.8

#Direction consistency
previous_direction_x = 0
previous_direction_y = 0

target_confidence = 0
prediction_confidence = 0


#Velocity consistency
previous_target_velocity_x = 0
previous_target_velocity_y = 0

# TARGET ACCELERATION
target_acceleration_x = 0
target_acceleration_y = 0
target_acceleration = 0
previous_target_speed = 0

# TARGET MOVEMENT STATE
target_movement_state = "STATIONARY"

# -------------------------
# YOLO
# -------------------------

yolo_model = YOLO("yolo11n.pt")

yolo_results = None
yolo_frame_counter = 0
yolo_interval = 30

# -------------------------
# START ARACHNE
# -------------------------

print("ARACHNE V0.1")
print("Starting camera...")


camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


if not camera.isOpened():

    print("ERROR: Could not access the camera.")

    exit()


print("Camera ONLINE")


# -------------------------
# FPS VARIABLES
# -------------------------

fps = 0

fps_start_time = time.time()

fps_frame_count = 0


# -------------------------
# MAIN LOOP
# -------------------------

while True:

    success, frame = camera.read()


    if not success:

        print("ERROR: Could not read camera frame.")

        break

    # ==================================================
    # YOLO RECOGNITION
    # ==================================================

    yolo_frame_counter += 1

    if yolo_frame_counter >= yolo_interval:

        yolo_results = yolo_model(
            frame,
            imgsz=320,
            verbose=False
        )

        yolo_frame_counter = 0
    # ==================================================
    # MOTION DETECTION
    # ==================================================

    gray_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    gray_frame = cv2.GaussianBlur(
        gray_frame,
        (21, 21),
        0
    )


    motion_detected = False

    largest_motion_area = 0

    largest_motion_box = None

    motion_candidates = []

    selected_candidate = None

    # -------------------------
    # COMPARE FRAMES
    # -------------------------

    if previous_frame is not None:

        difference = cv2.absdiff(
            previous_frame,
            gray_frame
        )


        _, threshold = cv2.threshold(
            difference,
            25,
            255,
            cv2.THRESH_BINARY
        )


        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )


        # -------------------------
        # FIND LARGEST MOTION
        # -------------------------

        for contour in contours:

            area = cv2.contourArea(contour)

            if area > 3000:

                x, y, w, h = cv2.boundingRect(contour)

                center_x = x + w // 2
                center_y = y + h // 2

                motion_candidates.append(
                    (
                        x,
                        y,
                        w,
                        h,
                        center_x,
                        center_y,
                        area
                    )
                )

                if area > largest_motion_area:

                    largest_motion_area = area

                    largest_motion_box = (
                        x,
                        y,
                        w,
                        h
                    )


        # ==================================================
        # MOTION FOUND
        # ==================================================

        if len(motion_candidates) > 0:


            # -------------------------
            # ACQUIRE TARGET
            # -------------------------

            if not target_locked:

                selected_candidate = max(
                    motion_candidates,
                    key=lambda candidate: candidate[6]
                )

            # -------------------------
            # TRACK TARGET
            # -------------------------

            else:

                closest_distance = maximum_target_jump
                best_score = float("inf")

                for candidate in motion_candidates:

                    candidate_x = candidate[4]
                    candidate_y = candidate[5]

                    candidate_width = candidate[2]
                    candidate_height = candidate[3]


                    # -------------------------
                    # DISTANCE
                    # -------------------------

                    distance = (
                        (
                            candidate_x - tracked_position[0]
                        ) ** 2
                        +
                        (
                            candidate_y - tracked_position[1]
                        ) ** 2
                    ) ** 0.5


                    if distance > maximum_target_jump:

                        continue


                    # -------------------------
                    # SIZE DIFFERENCE
                    # -------------------------

                    width_difference = abs(
                        candidate_width - target_width
                    )

                    height_difference = abs(
                        candidate_height - target_height
                    )


                    # -------------------------
                    # DIRECTION DIFFERENCE
                    # -------------------------

                    candidate_dx = (
                        candidate_x
                        -
                        tracked_position[0]
                    )

                    candidate_dy = (
                        candidate_y
                        -
                        tracked_position[1]
                    )

                    direction_difference = (
                        abs(
                            candidate_dx
                            -
                            previous_direction_x
                        )
                        +
                        abs(
                            candidate_dy
                            -
                            previous_direction_y
                        )
                    )
                    candidate_velocity_x = candidate_dx
                    candidate_velocity_y = candidate_dy

                    velocity_difference = (
                        abs(
                            candidate_velocity_x
                            -
                            previous_target_velocity_x
                        )
                        +
                        abs(
                            candidate_velocity_y
                            -
                            previous_target_velocity_y
                        )
                    )


                    # -------------------------
                    # COMBINED SCORE
                    # -------------------------

                    score = (
                        distance
                        +
                        (width_difference * 0.5)
                        +
                        (height_difference * 0.5)
                        +
                        (direction_difference * 0.3)
                        +
                        (velocity_difference * 0.4)
                    )


                    if score < best_score:

                        best_score = score

                        selected_candidate = candidate


            # -------------------------
            # TARGET CONFIDENCE
            # -------------------------



            if selected_candidate is not None:

                if target_locked and 'best_score' in locals():

                    target_confidence = int(
                        100 /
                        (
                            1 +
                            (best_score / 50)
                        )
                    )

                else:

                    target_confidence = 100


        # -------------------------
        # TARGET FOUND
        # -------------------------

        if selected_candidate is not None:

            motion_detected = True

            target_locked = True
            target_lost_frames = 0

            x, y, w, h, center_x, center_y, area = selected_candidate


            if target_tracking_start is None:

                target_tracking_start = time.time()

                    # -------------------------
                    # TARGET POSITION
                    # -------------------------

                if tracked_position is None:

                        tracked_position = (
                            center_x,
                            center_y
                        )

                else:

                        smooth_x = int(
                            tracked_position[0]
                            +
                            (
                                center_x
                                -
                                tracked_position[0]
                            )
                            *
                            position_smoothing
                        )

                        smooth_y = int(
                            tracked_position[1]
                            +
                            (
                                center_y
                                -
                                tracked_position[1]
                            )
                            *
                            position_smoothing
                        )

                        target_velocity_x = (
                            smooth_x
                            -
                            tracked_position[0]
                        )

                        target_velocity_y = (
                            smooth_y
                            -
                            tracked_position[1]
                        )

                        #-------------------------
                        # TARGET SPEED
                        #-------------------------
                        target_velocity_x = (
                            smooth_x
                            -
                            tracked_position[0]
                        )

                        target_velocity_y = (
                            smooth_y
                            -
                            tracked_position[1]
                        )

                        tracked_position = (
                            smooth_x,
                            smooth_y
                        )
                        # -------------------------
                        # TARGET ACCELERATION
                        # -------------------------
                
                        tracked_position = (
                            smooth_x,
                            smooth_y
                        )

                tracked_box = (
                    x,
                    y,
                    w,
                    h
                )

                target_width = w
                target_height = h
        
        # -------------------------
        # TARGET DISTANCE
        # -------------------------

        if target_height > 0:

            target_distance = 1000 / target_height

            if target_distance < 3:

                target_distance_status = "NEAR"

            elif target_distance < 7:

                target_distance_status = "MEDIUM"

            else:

                target_distance_status = "FAR"

        # -------------------------
            # DISTANCE TREND
            # -------------------------

            if target_distance_previous > 0:

                distance_difference = (
                    target_distance
                    -
                    target_distance_previous
                )

                if distance_difference < -0.15:

                    target_distance_trend = "APPROACHING"

                elif distance_difference > 0.15:

                    target_distance_trend = "MOVING AWAY"

                else:

                    target_distance_trend = "STABLE"


            target_distance_previous = target_distance

            # -------------------------
            # FIND CENTRE
            # -------------------------

            center_x = x + w // 2

            center_y = y + h // 2


            current_position = (
                center_x,
                center_y
            )


            # ==================================================
            # MOVEMENT ANALYSIS
            # ==================================================

            if len(position_history) > 0:

                previous_x, previous_y = position_history[-1]


                # -------------------------
                # DIFFERENCE IN POSITION
                # -------------------------

                dx = center_x - previous_x

                dy = center_y - previous_y

                #-------------------------
                # DIRECTION CONSISTENCY CHECK
                #-------------------------

                previous_direction_x = dx
                previous_direction_y = dy


                # -------------------------
                # CALCULATE SPEED
                # -------------------------

                speed = (
                    (dx ** 2 + dy ** 2) ** 0.5
                )


                # -------------------------
                # SMOOTH SPEED
                # -------------------------

                previous_target_speed = smoothed_speed

                smoothed_speed = (
                    smoothed_speed * 0.5
                ) + (
                    speed * 0.5
                )

                # -------------------------
                # TARGET ACCELERATION
                # -------------------------

                target_acceleration = (
                    smoothed_speed
                    -
                    previous_target_speed
                )

                # -------------------------
                # TARGET MOVEMENT STATE
                # -------------------------

                if smoothed_speed < 2:

                    target_movement_state = "STATIONARY"

                elif target_acceleration > 0.5:

                    target_movement_state = "ACCELERATING"

                elif target_acceleration < -0.5:

                    target_movement_state = "DECELERATING"

                else:

                    target_movement_state = "MOVING"
                # ==================================================
                # DIRECTION
                # ==================================================

                if abs(dx) > abs(dy):

                    if dx > 5:

                        direction = "RIGHT"

                    elif dx < -5:

                        direction = "LEFT"


                else:

                    if dy > 5:

                        direction = "DOWN"

                    elif dy < -5:

                        direction = "UP"


                # ==================================================
                # MOVEMENT CLASSIFICATION
                # ==================================================

                if smoothed_speed < 2:

                    movement_status = "STATIONARY"


                elif smoothed_speed < 6:

                    movement_status = "SLOW"


                elif smoothed_speed < 12:

                    movement_status = "MEDIUM"


                else:

                    movement_status = "FAST"


            # -------------------------
            # SAVE POSITION
            # -------------------------

            position_history.append(
                current_position
            )

            if len(position_history) > 20:
                position_history.pop(0)

                
            # ==================================================
            # MOTION VISUALIZATION
            # ==================================================

            # Motion box

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 255, 255),
                2
            )

            # ==================================================
            # TARGET LOCK RETICLE
            # ==================================================

            if target_locked:

                corner_length = 15


                # -------------------------
                # TOP LEFT
                # -------------------------

                cv2.line(
                    frame,
                    (x, y),
                    (x + corner_length, y),
                    (255, 255, 255),
                    2
                )

                cv2.line(
                    frame,
                    (x, y),
                    (x, y + corner_length),
                    (255, 255, 255),
                    2
                )


                # -------------------------
                # TOP RIGHT
                # -------------------------

                cv2.line(
                    frame,
                    (x + w, y),
                    (x + w - corner_length, y),
                    (255, 255, 255),
                    2
                )

                cv2.line(
                    frame,
                    (x + w, y),
                    (x + w, y + corner_length),
                    (255, 255, 255),
                    2
                )


                # -------------------------
                # BOTTOM LEFT
                # -------------------------

                cv2.line(
                    frame,
                    (x, y + h),
                    (x + corner_length, y + h),
                    (255, 255, 255),
                    2
                )

                cv2.line(
                    frame,
                    (x, y + h),
                    (x, y + h - corner_length),
                    (255, 255, 255),
                    2
                )


                # -------------------------
                # BOTTOM RIGHT
                # -------------------------

                cv2.line(
                    frame,
                    (x + w, y + h),
                    (x + w - corner_length, y + h),
                    (255, 255, 255),
                    2
                )

                cv2.line(
                    frame,
                    (x + w, y + h),
                    (x + w, y + h - corner_length),
                    (255, 255, 255),
                    2
                )


            # Centre point

            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (255, 255, 255),
                -1
            )

    # ==================================================
    # MOVEMENT TRAIL
    # ==================================================

    if len(position_history) > 1:

        trail_points = len(position_history)

        for i in range(1, trail_points):

            if movement_status == "STATIONARY":

                fade_ratio = i / trail_points

                thickness = max(
                    1,
                    int(2 * fade_ratio)
                )

            else:

                thickness = 2

            cv2.line(
                frame,
                position_history[i - 1],
                position_history[i],
                (255, 255, 255),
                thickness
            )

    # ==================================================
    # PROJECTED MOVEMENT VECTOR
    # ==================================================

    if target_locked and tracked_position is not None:

        vector_start_x = tracked_position[0]
        vector_start_y = tracked_position[1]

        vector_end_x = int(
            vector_start_x
            +
            (
                target_velocity_x
                *
                vector_scale
            )
        )

        vector_end_y = int(
            vector_start_y
            +
            (
                target_velocity_y
                *
                vector_scale
            )
        )

        vector_end_x = max(
            0,
            min(639, vector_end_x)
        )

        vector_end_y = max(
            0,
            min(479, vector_end_y)
        )

        cv2.arrowedLine(
            frame,
            (
                vector_start_x,
                vector_start_y
            ),
            (
                vector_end_x,
                vector_end_y
            ),
            (255, 255, 255),
            2,
            tipLength=0.25
        )

    # ==================================================
    # NO MOTION / TARGET LOSS
    # ==================================================

    if motion_detected:

        no_motion_frames = 0
        prediction_confidence = 100
        target_lost_frames = 0
        tracking_prediction_frames = 0

    else:

        no_motion_frames += 1

        target_lost_frames += 1

            # Gradually reduce speed

        smoothed_speed *= 0.95
    

            # -------------------------
            # TARGET PERSISTENCE
            # -------------------------

        if target_locked and tracked_position is not None:

                tracking_prediction_frames += 1

                target_velocity_x *= prediction_velocity_decay

                target_velocity_y *= prediction_velocity_decay


                if no_motion_frames > 15:

                    smoothed_speed = 0

                    movement_status = "STATIONARY"

                    direction = "STATIONARY"


        # Lose target after 15 frames

        if tracking_prediction_frames > maximum_prediction_frames:

            target_locked = False

            tracked_position = None
            tracked_box = None

            target_velocity_x = 0
            target_velocity_y = 0

            tracking_prediction_frames = 0

            target_tracking_start = None
            target_tracking_time = 0


# ==================================================
# TARGET PREDICTION
# ==================================================

        if not motion_detected and target_locked:

            if tracking_prediction_frames <= maximum_prediction_frames:

                tracked_x, tracked_y = tracked_position

                predicted_x = (
                    tracked_x +
                    target_velocity_x
                )

                predicted_y = (
                    tracked_y +
                    target_velocity_y
                )

                predicted_x = max(
                    0,
                    min(639, predicted_x)
                )

                predicted_y = max(
                    0,
                    min(479, predicted_y)
                )

                tracked_position = (
                    int(predicted_x),
                    int(predicted_y)
                )
            prediction_confidence = int(
        max(
            0,
            100 - (
                tracking_prediction_frames
                *
                6
            )
        )
    )

    #TARGET PREDICTION HUD TEXT

    if not motion_detected and target_locked:

        cv2.putText(
            frame,
            "TARGET PREDICTING",
            (20, 390),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    # Save current frame

    previous_frame = gray_frame


    # ==================================================
    # FPS
    # ==================================================

    fps_frame_count += 1


    current_time = time.time()


    if current_time - fps_start_time >= 1:

        fps = fps_frame_count

        fps_frame_count = 0

        fps_start_time = current_time


    if target_locked and target_tracking_start is not None:

        target_tracking_time = (
            time.time() - target_tracking_start
        )

    # ==================================================
    # ARACHNE HUD
    # ==================================================

    # -------------------------
    # TITLE
    # -------------------------

    cv2.putText(
        frame,
        "ARACHNE V0.1",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # -------------------------
    # FPS
    # -------------------------

    cv2.putText(
        frame,
        f"FPS: {fps}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # -------------------------
    # VISION
    # -------------------------

    cv2.putText(
        frame,
        "VISION: ONLINE",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # -------------------------
    # SYSTEM
    # -------------------------

    cv2.putText(
        frame,
        "SYSTEM: NORMAL",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # -------------------------
    # MOTION
    # -------------------------

    if motion_detected:

        cv2.putText(
            frame,
            "MOTION DETECTED",
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "MOTION: CLEAR",
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


    # -------------------------
    # DIRECTION
    # -------------------------

    cv2.putText(
        frame,
        f"DIRECTION: {direction}",
        (20, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # -------------------------
    # SPEED
    # -------------------------

    cv2.putText(
        frame,
        f"SPEED: {smoothed_speed:.1f} PX/FRAME",
        (20, 245),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # -------------------------
    # MOVEMENT
    # -------------------------

    cv2.putText(
        frame,
        f"MOVEMENT: {movement_status}",
        (20, 280),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # -------------------------
    # TARGET STATUS
    # -------------------------

    if target_locked:

        target_text = f"TARGET {target_id}: LOCKED"

    else:

        target_text = "TARGET: SEARCHING"


    cv2.putText(
        frame,
        target_text,
        (20, 315),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    #-------------------------
    # TARGET CONFIDENCE
    #-------------------------
    cv2.putText(
        frame,
        f"CONFIDENCE: {target_confidence}%",
        (20, 340),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

#Tracking Timing
    cv2.putText(
        frame,
        f"TRACK TIME: {target_tracking_time:.1f}s",
        (20, 365),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    # ==================================================
    # TARGETING RETICLE
    # ==================================================

    center_x = 320
    center_y = 240

    box_size = 80

    left = center_x - box_size
    right = center_x + box_size

    top = center_y - box_size
    bottom = center_y + box_size

    corner_length = 20


    # -------------------------
    # TOP LEFT
    # -------------------------

    cv2.line(
        frame,
        (left, top),
        (left + corner_length, top),
        (255, 255, 255),
        2
    )

    cv2.line(
        frame,
        (left, top),
        (left, top + corner_length),
        (255, 255, 255),
        2
    )


    # -------------------------
    # TOP RIGHT
    # -------------------------

    cv2.line(
        frame,
        (right, top),
        (right - corner_length, top),
        (255, 255, 255),
        2
    )

    cv2.line(
        frame,
        (right, top),
        (right, top + corner_length),
        (255, 255, 255),
        2
    )


    # -------------------------
    # BOTTOM LEFT
    # -------------------------

    cv2.line(
        frame,
        (left, bottom),
        (left + corner_length, bottom),
        (255, 255, 255),
        2
    )

    cv2.line(
        frame,
        (left, bottom),
        (left, bottom - corner_length),
        (255, 255, 255),
        2
    )


    # -------------------------
    # BOTTOM RIGHT
    # -------------------------

    cv2.line(
        frame,
        (right, bottom),
        (right - corner_length, bottom),
        (255, 255, 255),
        2
    )

    cv2.line(
        frame,
        (right, bottom),
        (right, bottom - corner_length),
        (255, 255, 255),
        2
    )


    # -------------------------
    # CENTER CROSSHAIR
    # -------------------------

    cv2.line(
        frame,
        (center_x - 10, center_y),
        (center_x + 10, center_y),
        (255, 255, 255),
        1
    )

    cv2.line(
        frame,
        (center_x, center_y - 10),
        (center_x, center_y + 10),
        (255, 255, 255),
        1
    )
    
    #=====================================
    #PREDICTION CONFIDENCE
    #=====================================
    cv2.putText(
    frame,
    f"PREDICTION: {prediction_confidence}%",
    (20, 415),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (255, 255, 255),
    2
)

    #Target Distance 
    cv2.putText(
        frame,
        f"DISTANCE: {target_distance_status}",
        (20, 440),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    cv2.putText(
    frame,
    f"TREND: {target_distance_trend}",
    (20, 465),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (255, 255, 255),
    2
)

    # ==================================================
    # INFORMATION PANEL
    # ==================================================

    panel_x = 450
    panel_y = 30


    cv2.putText(
        frame,
        "ARACHNE SYSTEM",
        (panel_x, panel_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    cv2.putText(
        frame,
        "CAMERA   ONLINE",
        (panel_x, panel_y + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )


    cv2.putText(
        frame,
        "VISION   READY",
        (panel_x, panel_y + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )


    cv2.putText(
        frame,
        "AI       STANDBY",
        (panel_x, panel_y + 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    cv2.putText(
    frame,
    f"ACCEL   {target_acceleration:.1f}",
    (panel_x, panel_y + 85),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.45,
    (255, 255, 255),
    1
)
    cv2.putText(
        frame,
        f"STATE   {target_movement_state}",
        (panel_x, panel_y + 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )


    # ==================================================
    # DISPLAY
    # ==================================================

    cv2.imshow(
        "ARACHNE V0.1",
        frame
    )


    # -------------------------
    # QUIT
    # -------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==================================================
# SHUTDOWN
# ==================================================

camera.release()

cv2.destroyAllWindows()

print("\nARACHNE OFFLINE.")