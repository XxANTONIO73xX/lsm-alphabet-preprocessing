import os
import cv2
import argparse
import mediapipe as mp

from tqdm import tqdm

mp_hands = mp.solutions.hands

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_hand_bbox(hand_landmarks, width, height):

    xs = [lm.x * width for lm in hand_landmarks.landmark]
    ys = [lm.y * height for lm in hand_landmarks.landmark]

    x_min = int(min(xs))
    y_min = int(min(ys))

    x_max = int(max(xs))
    y_max = int(max(ys))

    return x_min, y_min, x_max, y_max


def compute_centroid(bbox):

    x1, y1, x2, y2 = bbox

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    return cx, cy


def crop_from_center(frame, center, crop_size):

    h, w = frame.shape[:2]

    cx, cy = center

    half = crop_size // 2

    x1 = max(cx - half, 0)
    y1 = max(cy - half, 0)

    x2 = min(cx + half, w)
    y2 = min(cy + half, h)

    crop = frame[y1:y2, x1:x2]

    pad_top = max(0, half - cy)
    pad_bottom = max(0, (cy + half) - h)

    pad_left = max(0, half - cx)
    pad_right = max(0, (cx + half) - w)

    crop = cv2.copyMakeBorder(
        crop,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    crop = cv2.resize(crop, (crop_size, crop_size))

    return crop

def process_video(
    video_path,
    output_root,
    hands,
    crop_size
):

    video_name = os.path.splitext(
        os.path.basename(video_path)
    )[0]

    output_dir = os.path.join(output_root, video_name)

    ensure_dir(output_dir)

    cap = cv2.VideoCapture(video_path)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    last_center = None

    for frame_idx in tqdm(
        range(total_frames),
        desc=video_name
    ):

        ret, frame = cap.read()

        if not ret:
            break

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb)

        if results.multi_hand_landmarks:

            hand_landmarks = results.multi_hand_landmarks[0]

            bbox = get_hand_bbox(
                hand_landmarks,
                width,
                height
            )

            center = compute_centroid(bbox)

            last_center = center

        if last_center is not None:

            crop = crop_from_center(
                frame,
                last_center,
                crop_size
            )

        else:

            crop = cv2.resize(
                frame,
                (crop_size, crop_size)
            )


        output_name = (
            f"{video_name}_{frame_idx:06d}.jpg"
        )

        output_path = os.path.join(
            output_dir,
            output_name
        )

        cv2.imwrite(
            output_path,
            crop,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

    cap.release()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input video directory"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output crop directory"
    )

    parser.add_argument(
        "--crop-size",
        type=int,
        default=360,
        help="Crop size"
    )

    parser.add_argument(
        "--max-hands",
        type=int,
        default=1
    )

    args = parser.parse_args()

    # =====================================================
    # MEDIAPIPE INIT
    # =====================================================

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=args.max_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    valid_ext = (
        ".mp4",
        ".mov",
        ".avi",
        ".mkv"
    )

    videos = []

    for root, _, files in os.walk(args.input):

        for file in files:

            if file.lower().endswith(valid_ext):

                videos.append(
                    os.path.join(root, file)
                )

    print(f"Videos found: {len(videos)}")

    for video_path in videos:

        process_video(
            video_path=video_path,
            output_root=args.output,
            hands=hands,
            crop_size=args.crop_size
        )

    print("Processing complete.")


if __name__ == "__main__":
    main()