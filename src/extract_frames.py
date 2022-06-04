import argparse
import os
import cv2
import multiprocessing


def extract(video, frames_dir, video_dir):
    """
    function extracts frames from vidoe files

    video: mp4 file name of video
    frames_dir: path to frame output directory
    video_dir: path to directory containing mp4 videos
    """
    out_dir = os.path.join(frames_dir, video.split('.')[0].zfill(3), 'rgb')
    video_path = os.path.join(video_dir, video)
    if not os.path.isdir(out_dir) or len(os.listdir(out_dir)) != 1800:

        video_cap = cv2.VideoCapture(video_path)
        os.makedirs(out_dir, exist_ok=True)

        count = 1
        success = True

        print('Elaborating: ' + video_path)
        while success:
            success, image = video_cap.read()

            filename = os.path.join(out_dir, str(count).zfill(4) + '.jpg')
            cv2.imwrite(filename, image)
            count += 1


def main():
    parser = argparse.ArgumentParser(description='Get frames from videos')
    parser.add_argument(
        '--video_dir', help='Directory hosting videos', required=True)
    parser.add_argument('--out_dir', help='Directory of output', required=True)

    args = parser.parse_args()

    video_dir = args.video_dir
    out_dir = args.out_dir

    frames_dir = os.path.join(out_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    video_list = os.listdir(video_dir)
    pool = multiprocessing.Pool(processes=20)
    print("start", out_dir, video_dir)
    [pool.apply_async(extract, (video, frames_dir, video_dir))
     for video in video_list]

    pool.close()
    pool.join()


if __name__ == '__main__':
    main()