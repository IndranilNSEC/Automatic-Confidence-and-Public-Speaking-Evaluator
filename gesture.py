import tensorflow as tf
import numpy as np
import cv2
import json

def body_gesture(vid_path):
    interpreter = tf.lite.Interpreter(model_path='./Model\\gesture.tflite')
    interpreter.allocate_tensors()

    EDGES = {
        (0, 1): 'm',
        (0, 2): 'c',
        (1, 3): 'm',
        (2, 4): 'c',
        (0, 5): 'm',
        (0, 6): 'c',
        (5, 7): 'm',
        (6, 8): 'c',
        (7, 9): 'm',
        (8, 10): 'c',
        (5, 6): 'y',
        (5, 11): 'm',
        (6, 12): 'c',
        (11, 12): 'y',
        (11, 13): 'm',
        (12, 14): 'c',
        (13, 15): 'm',
        (14, 16): 'c'
    }

    def draw_keypoints(frame, keypoints, confidence_threshold):
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

        for kp in shaped:
            ky, kx, kp_conf = kp
            if kp_conf > confidence_threshold:
                cv2.circle(frame, (int(kx), int(ky)), 6, (0, 255, 0), -1)

    def draw_connections(frame, keypoints, edges, confidence_threshold):
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

        for edge, color in edges.items():
            p1, p2 = edge
            if p1 < len(shaped) and p2 < len(shaped):
                y1, x1, c1 = shaped[p1]
                y2, x2, c2 = shaped[p2]
                if (c1 > confidence_threshold) and (c2 > confidence_threshold):
                    color_map = {
                        'm': (0, 255, 0),
                        'c': (255, 0, 0),
                        'y': (0, 255, 255)
                    }
                    line_color = color_map.get(color, (0, 0, 255))
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), line_color, 2)

    def is_keypoint_visible(keypoint):
        return keypoint[2] > 0.7  # Threshold of Detecting Landmarks

    def smooth_keypoints(previous, current, alpha=0.5):
        if previous is None:
            return current
        return previous * alpha + current * (1 - alpha)

    def detect_open_pose(keypoints):
        if len(keypoints) >= 17:
            shoulders_back = (keypoints[5][1] < keypoints[6][1]) or not is_keypoint_visible(keypoints[5]) or not is_keypoint_visible(keypoints[6])
            arms_open = ((keypoints[7][0] < keypoints[5][0] and keypoints[8][0] > keypoints[6][0]) or
                          (not is_keypoint_visible(keypoints[7]) or not is_keypoint_visible(keypoints[8])))
            head_high = (keypoints[0][1] < keypoints[5][1] and keypoints[0][1] < keypoints[6][1]) or not is_keypoint_visible(keypoints[0])
            return shoulders_back and arms_open and head_high
        return False

    def detect_closed_pose(keypoints):
        if len(keypoints) >= 17:
            left_arm_cross = (keypoints[5][0] > keypoints[11][0]) or not is_keypoint_visible(keypoints[5]) or not is_keypoint_visible(keypoints[11])
            right_arm_cross = (keypoints[6][0] < keypoints[12][0]) or not is_keypoint_visible(keypoints[6]) or not is_keypoint_visible(keypoints[12])
            head_down = (keypoints[0][1] > keypoints[5][1] and keypoints[0][1] > keypoints[6][1]) or not is_keypoint_visible(keypoints[0])
            return left_arm_cross and right_arm_cross and head_down
        return False

    def detect_relaxed_pose(keypoints):
        if len(keypoints) >= 17:
            arms_relaxed = (keypoints[5][1] > 0.5 and keypoints[6][1] > 0.5) or not is_keypoint_visible(keypoints[5]) or not is_keypoint_visible(keypoints[6])
            weight_even = abs(keypoints[11][0] - keypoints[12][0]) < 0.1 or (not is_keypoint_visible(keypoints[11]) and not is_keypoint_visible(keypoints[12]))
            head_neutral = abs(keypoints[0][1] - (keypoints[5][1] + keypoints[6][1]) / 2) < 0.1 or not is_keypoint_visible(keypoints[0])
            return arms_relaxed and weight_even and head_neutral
        return False

    def detect_in_control_pose(keypoints):
        if len(keypoints) >= 17:
            feet_stance = (abs(keypoints[11][0] - keypoints[12][0]) > 0.2) or not is_keypoint_visible(keypoints[11]) or not is_keypoint_visible(keypoints[12])
            hands_on_hips = (keypoints[5][0] < keypoints[11][0] and keypoints[6][0] > keypoints[12][0]) or not is_keypoint_visible(keypoints[5]) or not is_keypoint_visible(keypoints[6])
            head_up = (keypoints[0][1] < keypoints[5][1] and keypoints[0][1] < keypoints[6][1]) or not is_keypoint_visible(keypoints[0])
            return feet_stance and hands_on_hips and head_up
        return False

    cap = cv2.VideoCapture(vid_path)
    previous_keypoints = None
    all_keypoints = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize image to 256x256 and convert to float32
        img = cv2.resize(frame, (256, 256))
        input_image = img.astype(np.float32)
        # input_image = (img * 255).astype(np.uint8)   #For 4.tflite model - Different Input size
        
        # Set input tensor
        input_details = interpreter.get_input_details()
        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(input_image, axis=0))

        # Run inference
        interpreter.invoke()

        # Get output tensor and reshape
        output_details = interpreter.get_output_details()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

        # Normalize keypoints to [0.0, 1.0]
        keypoints_with_scores = np.clip(keypoints_with_scores, 0.0, 1.0)

        # Smooth the keypoints
        smoothed_keypoints = smooth_keypoints(previous_keypoints, keypoints_with_scores[0, 0])


        keypoint_confidences = [kp[2] for kp in smoothed_keypoints]
        if not any(conf > 0.4 for conf in keypoint_confidences):  # Threshold of No pose Detection
            cv2.putText(frame, "No Pose Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Rendering 
            all_keypoints.append(smoothed_keypoints)
            draw_connections(frame, smoothed_keypoints, EDGES, 0.4)
            draw_keypoints(frame, smoothed_keypoints, 0.4)

            # Pose detection
            if detect_open_pose(smoothed_keypoints):
                cv2.putText(frame, "Open Pose", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif detect_closed_pose(smoothed_keypoints):
                cv2.putText(frame, "Closed Pose", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif detect_relaxed_pose(smoothed_keypoints):
                cv2.putText(frame, "Relaxed Pose", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            elif detect_in_control_pose(smoothed_keypoints):
                cv2.putText(frame, "In Control Pose", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # Debugging: Print out the confidence scores of the keypoints
        # print("Keypoint confidences:", keypoint_confidences)
        
        cv2.imshow('MoveNet Thunder', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        # Update the previous keypoints for the next frame
        previous_keypoints = smoothed_keypoints

    cap.release()
    cv2.destroyAllWindows()

    def detect_pose(keypoints):

        if detect_open_pose(keypoints):
            return "Open Pose"
        elif detect_closed_pose(keypoints):
            return "Closed Pose"
        elif detect_relaxed_pose(keypoints):
            return "Relaxed Pose"
        elif detect_in_control_pose(keypoints):
            return "In Control Pose"
        return "No Pose Detected"

    if all_keypoints:
        average_keypoints = np.mean(all_keypoints, axis=0)
        print("Average Keypoints:")
        print(average_keypoints)

        average_pose_result = detect_pose(average_keypoints)
        print(f"Estimated Average Pose: {average_pose_result}")
    else:
        average_pose_result = "No Pose Detected"
        
    # Create a dictionary containing the final gesture data
    final_gesture_data = {'dominant_gesture': average_pose_result}
        
    # Wrap the final gesture data inside a dictionary with the key 'gesture_result'
    result = {'gesture_result': final_gesture_data}
    
    # Convert the dictionary to JSON format
    json_data_recorded = json.dumps(result)
    
    print(json_data_recorded)
    
    return json_data_recorded

#body_gesture('Videos\VID-20240911-WA0007.mp4')