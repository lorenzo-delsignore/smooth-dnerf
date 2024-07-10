import argparse
import os

import cv2


def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    num_frames = fps * 2
    num_extracted_frames = min(10, num_frames)
    step_size = max(1, num_frames // num_extracted_frames)
    os.makedirs(output_folder, exist_ok=True)
    for index in range(0, num_frames, step_size):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(output_folder, f"frame_{index}.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"Frame {index} saved to {frame_path}")
        else:
            print(f"Error reading frame {index}")
    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frames from a video for animation evaluation"
    )
    parser.add_argument("-video_path", type=str, help="Path to the input video file")
    parser.add_argument(
        "-output_folder", type=str, help="Path to the output folder to save frames"
    )
    args = parser.parse_args()
    extract_frames(args.video_path, args.output_folder)
