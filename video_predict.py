import os
import os.path as osp
import cv2
from PIL import Image
import streamlit as st


def runVideo(model, video, vdo_view, warn):
    try:
        os.makedirs("./data/video_frames")
    except FileExistsError:
        print("File already exists")
    try:
        os.makedirs("./data/video_output")
    except FileExistsError:
        print("File already exists")
    video_name = osp.basename(video)
    outputpath = osp.join('data/video_output', video_name)

    # Create A Dir to save Video Frames
    os.makedirs('data/video_frames', exist_ok=True)
    frames_dir = osp.join('data/video_frames', video_name)
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_count = 0
    with st.spinner(text="Predicting..."):
        warn.warning(
            'This is realtime prediction, If you wish to download the final prediction result wait until the process done.', icon="➡")
        while True:
            frame_count += 1
            ret, frame = cap.read()
            if ret == False:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = model(frame)
            print(f"Processing frame {frame_count}\n")
            print(result)
            #result.render() //draws bounding box and labels by default
            boxes = result.xyxy[0].numpy()
            labels = result.names
            class_color = [[0,255,0],[255,0,0]]
            for box,label in zip(boxes,labels):
                x1,y1,x2,y2,confidence_score,class_id = box.astype(int)
                if labels[class_id]=='No Mask' or labels[class_id]=='No Gloves' or labels[class_id]=='No Vest':
                    box_color = class_color[1]
                else:
                    box_color = class_color[0] 
                cv2.rectangle(frame,(x1,y1),(x2,y2),box_color,2)
                cv2.putText(frame,labels[class_id],(x1,y1-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,box_color,2)
            image = Image.fromarray(frame)
            vdo_view.image(image, caption='Current Model Prediction(s)')
            image.save(osp.join(frames_dir, f'{frame_count}.jpg'))
        cap.release()
        os.system(f'ffmpeg -framerate 1 -pattern_type glob -i {frames_dir}/%d.jpg -c:v libx264 -r 30 -pix_fmt yuv420p {outputpath}')

    # Display Video
    '''
    frame_width = int(width)
    frame_height = int(height)
    frame_files = os.listdir(frames_dir)
    frame_files.sort(key=lambda x: os.stat(os.path.join(frames_dir, x)).st_ctime)
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # You can use other codecs as well, such as 'XVID'
    video = cv2.VideoWriter(outputpath, fourcc, video_fps, (frame_width, frame_height))
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            video.write(frame)
    video.release()
    '''
    output_video = open(outputpath, mode='rb')
    output_video_bytes = output_video.read()
    st.write("Model Prediction")
    st.video(output_video_bytes)
    vdo_view.empty()
    warn.empty()
