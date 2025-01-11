import os
import time
import cv2
import numpy as np
from pyorbbecsdk import *
import open3d as o3d

ESC_KEY = 27
VIDEO_FILE = "12345.avi"  # Output video file
FRAME_RATE = 50  # Desired frame rate
VIDEO_RESOLUTION = (640, 480)  # Adjust based on depth frame resolution

def main():
    # Initialize pipeline
    pipeline = Pipeline()
    config = Config()
    device = pipeline.get_device()
    depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

    if depth_profile_list is None:
        print("No proper depth profile available. Exiting...")
        return

    depth_profile = depth_profile_list.get_default_video_stream_profile()
    config.enable_stream(depth_profile)
    pipeline.start(config)

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
    video_writer = cv2.VideoWriter(VIDEO_FILE, fourcc, FRAME_RATE, VIDEO_RESOLUTION)

    print(f"Recording depth video to {VIDEO_FILE}... Press 'q' or ESC to stop.")
    
    try:
        while True:
            frames = pipeline.wait_for_frames(1)
            if frames is None:
                continue

            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                continue

            # Extract depth data
            width = depth_frame.get_width()
            height = depth_frame.get_height()
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))

            # Normalize and convert depth data to 8-bit for video
            depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

            # Resize if necessary to match the desired resolution
            if (width, height) != VIDEO_RESOLUTION:
                depth_image = cv2.resize(depth_image, VIDEO_RESOLUTION)

            # Write frame to video file
            video_writer.write(depth_image)

            # Display depth image for debugging
            cv2.imshow("Depth Video", depth_image)

            # Check for exit key
            key = cv2.waitKey(1)
            if key == ord('q') or key == ESC_KEY:
                break

    except KeyboardInterrupt:
        print("\nRecording stopped by user.")

    finally:
        # Cleanup resources
        pipeline.stop()
        video_writer.release()
        cv2.destroyAllWindows()
        print(f"Depth video saved as {VIDEO_FILE}")

if __name__ == "__main__":
    main()
