# from ultralytics import YOLO
#
# model = YOLO('r_int_g.pt')
#
# results = model(source=0, show=True, conf=0.4, save=True)


from ultralytics import YOLO
import cv2

# Load the model
model = YOLO('r_int_g.pt')

# Define the class index to label mapping
class_map = {2: 'red_tomato', 0: 'green_tomato', 1: 'intermediate_tomato'}

# Initialize cumulative counts
total_red_count = 0
total_green_count = 0
total_intermediate_count = 0

# Flag to check if a tomato is currently in the frame
tomato_in_frame = False

# Start the video capture
cap = cv2.VideoCapture(0)

# Real-time detection loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection on the current frame
    results = model.predict(source=frame, conf=0.4)

    # Track whether a tomato is detected in this frame
    current_frame_has_tomato = False

    # Process results and draw bounding boxes
    for result in results:
        for box in result.boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_index = int(box.cls[0])
            label = class_map.get(class_index, "unknown")

            # Set the flag to indicate a tomato is detected
            current_frame_has_tomato = True

            # Only count if no tomato was previously in the frame
            if not tomato_in_frame:
                if label == 'red_tomato':
                    total_red_count += 1
                elif label == 'green_tomato':
                    total_green_count += 1
                elif label == 'intermediate_tomato':
                    total_intermediate_count += 1

                # Set tomato_in_frame to True, so we don’t count until it exits
                tomato_in_frame = True

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
            cv2.putText(frame, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # If no tomato is detected in the current frame, reset the flag
    if not current_frame_has_tomato:
        tomato_in_frame = False

    # Display cumulative counts on the frame
    cv2.putText(frame, f"Total Red: {total_red_count}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"Total Green: {total_green_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Total Intermediate: {total_intermediate_count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Show the frame with detections
    cv2.imshow("Tomato Detection", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
