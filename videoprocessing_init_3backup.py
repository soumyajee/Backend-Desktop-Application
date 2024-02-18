from architectures import fornet, weights
from PIL import Image, ImageChops
from torchvision.transforms import ToPILImage
from io import BytesIO
from skimage import io
import imageio
from PIL import Image, ImageSequence
from scipy import ndimage
from skimage import transform
import numpy as np
import requests
import base64
import shutil
from typing_extensions import Concatenate

from isplutils import utils
from blazeface import FaceExtractor, BlazeFace, VideoReader
import os
import torch
import re
from gradcam import GradCAM
import matplotlib.pyplot as plt
from scipy.special import expit
import argparse
import cv2
import sys
import threading
import logging
import warnings
import json
import time
import datetime


threshold = 0.3
warnings.filterwarnings("ignore")
# Filter out the specific warning messages
warnings.filterwarnings("ignore", category=UserWarning, message="Failed to load image Python extension:")
warnings.filterwarnings("ignore", category=UserWarning, message="Unable to retrieve source for @torch.jit._overload function:")

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.DEBUG)
logging.basicConfig(filename='test1.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')
logging.getLogger('matplotlib.font_manager').setLevel('WARNING')
logging.basicConfig()
logging.getLogger('PIL').setLevel(logging.WARNING)
 # Load the model
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logging.info(f"{current_datetime} -Application started")
"""
Choose an architecture between
- EfficientNetB4
- EfficientNetB4ST
- EfficientNetAutoAttB4
- EfficientNetAutoAttB4ST
- Xception
"""
# Load the configuration file
with open('config.json', 'r') as f:
    config = json.load(f)

# Extract the model file path and other settings from the config
model_path = config['model_path']
face_policy = config['face_policy']
face_size = config['face_size']
frames_per_video = config['frames_per_video']
# Log the configuration parameters
logging.info(f"{current_datetime} - Face Policy: {face_policy}")
logging.info(f"{current_datetime} - Face Size: {face_size}")
logging.info(f"{current_datetime} - Frames Per Video: {frames_per_video}")
net_model = 'EfficientNetAutoAttB4'
#net_model = 'EfficientNetGenAutoAtt'
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#device = torch.device('cpu')
net = getattr(fornet, net_model)().eval().to(device)
# net = getattr(fornet, net_model)
# print(net.keys())
#torch.load(model_path)
# print(torch.load(model_path).keys())
# print (torch.load(model_path).keys())
#net.load_state_dict(torch.load(model_path,map_location='cpu'))['net']
net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['net'])
# net.eval().to(device)
logging.info(f"{current_datetime} - Loading the model from: {model_path}")
transf = utils.get_transformer(
    face_policy, face_size, net.get_normalizer(), train=False)
facedet = BlazeFace().to(device)
facedet.load_weights("./blazeface/blazeface.pth")
logging.info("Loading anchors at %s", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
facedet.load_anchors("./blazeface/anchors.npy")
def get_media_details(vidpath):
    video_size_mb = os.path.getsize(vidpath) / (1024 * 1024)  # Convert to megabytes
    video_size_str = f"{video_size_mb:.2f} MB"
    video_capture = cv2.VideoCapture(vidpath)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    resolution = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_codec = int(video_capture.get(cv2.CAP_PROP_FOURCC))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps
       # Log video details including size
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    processing_time = time.time() - time.time()  # You may need to fix the computation of processing_time
    LOGGER.info(f"{current_datetime} -Video details: FPS={fps}, Resolution={resolution}, Video Codec={video_codec}, Total Frames={total_frames}, Duration={duration_seconds} seconds, Size={video_size_str} ({processing_time} seconds)")

    # Close the video file
    video_capture.release()
    return fps,resolution,video_codec,total_frames,duration_seconds,video_size_mb
def core_processing_unit(vidpath):
    ## It will return the result in list format
    # Check if the video file exists
     # Check if the video file exists
     # List of acceptable video file extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv','.webm']  # Add more if needed 
    file_extension = os.path.splitext(vidpath)[1].lower()
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"{current_datetime} -Video processing started")
    start_time = time.time()
    end_time = time.time()
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    processing_time = end_time - start_time 
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y_%m_%d__%H_%M_%S')
    # Create a new directory with the current date and time
    results_dir = os.path.join('results', timestamp)
    os.makedirs(results_dir)  
    logging.info(f"{current_datetime} -Checking video file found or not ({processing_time} seconds)")
    # Call get_media_details to get video details
    if not os.path.isfile(vidpath):
         # Construct the results_details dictionary with error information
        results_details = {
            'video_details': {

            },
            'DetectionThreshold': 0,
            'confidence_percentage': 0,
            'result': "",
            'error_message': "Video file path not found",
            'video_path': vidpath,
            'heatmap': "",
            'processed_faces': "",
            'graph': ""
        }
        LOGGER.info("Video FIle not found") 
        results_details_json = json.dumps(results_details, indent=4).replace('\n', '').replace(' ', '')
        video_details_file_path = os.path.join(results_dir, 'results.json')
        with open(video_details_file_path, 'w') as f:
            f.write(results_details_json)  
        return video_details_file_path
    # Check if the file extension is in the list of acceptable video extensions
    elif file_extension not in video_extensions:
         # Construct the results_details dictionary with error information
        results_details = {
            'video_details': {},
            'DetectionThreshold': 0,
            'confidence_percentage': 0,
            'result': "",
            'error_message': "Invalid video file format",
            'video_path': vidpath,
            'heatmap': "",
            'processed_faces': "",
            'graph': ""
        }
        LOGGER.info("Invalid video file format") 
        results_details_json = json.dumps(results_details, indent=4).replace('\n', '').replace(' ', '')
        video_details_file_path = os.path.join(results_dir, 'results.json')
        with open(video_details_file_path, 'w') as f:
            f.write(results_details_json)  
        return video_details_file_path
    else:
        LOGGER.info("Valid Video Files and ready for processing")    
    ## It will handle the video if it contains video faces or not if not it returns no faces and invalid video
    try:  
          ## This function is bascally read the video
        videoreader = VideoReader(verbose=False)
        video_read_fn = lambda x: videoreader.read_frames(x, num_frames=frames_per_video)
        face_extractor = FaceExtractor(video_read_fn=video_read_fn,facedet=facedet)
        ## This function is basically extarct the video from videopath
        vid_faces = face_extractor.process_video(vidpath)
        print(type(vid_faces))
        logging.info(f"{current_datetime} -checking faces available or not({processing_time} seconds)")
        # Call get_media_details to get video details
        fps, resolution, video_codec, total_frames, duration_seconds,video_size_mb = get_media_details(vidpath)
        if len(vid_faces) == 0:
            results_details = {
            'video_details': {
            'FPS': int(fps),
            'Resolution': resolution,
            'VideoCodec': video_codec,
            'DurationSeconds': int(duration_seconds),
            },
            'DetectionThreshold': 0,
            'confidence_percentage': 0,
            'result': "",
            'error_message': "No Faces Found in videos",
            'video_path': vidpath,
            'heatmap': "",
            'processed_faces': "",
            'graph': ""
        }           
            LOGGER.info("No Faces Found in videos")
            results_details_json = json.dumps(results_details, indent=4).replace('\n', '').replace(' ', '')
            video_details_file_path = os.path.join(results_dir, 'results.json')
            with open(video_details_file_path, 'w') as f:
             f.write(results_details_json)  
            return video_details_file_path
        
         # Save the processed frames into a local directory.
           # Handle the case where the video has no faces
        faces_directory = "processed_faces"
        if not os.path.exists(faces_directory):            
                    os.makedirs(faces_directory)
        for i, frame in enumerate(vid_faces):
                    frame_faces = frame['faces']
                    for j, face in enumerate(frame_faces):
                        face_path = os.path.join(faces_directory, f"frame_{i}_face_{j}.jpg")
                        cv2.imwrite(face_path, face)        
         # Iterate over the processed faces and add them to the scene
        for i, frame in enumerate(vid_faces):
            frame_faces = frame['faces']
            for j, face in enumerate(frame_faces):
                face_path = os.path.join(faces_directory, f"frame_{i}_face_{j}.jpg")   
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                cv2.imwrite(face_path, face)
        logging.info(f'{current_datetime} - check if video file is less than thirty frames({processing_time} seconds)')   
        # Call get_media_details to get video details
        fps, resolution, video_codec, total_frames, duration_seconds,video_size_mb = get_media_details(vidpath) 
        if  len(vid_faces) < 32:
            results_details = {
            'video_details': {
            'FPS': int(fps),
            'Resolution': resolution,
            'VideoCodec': video_codec,
            'DurationSeconds': int(duration_seconds),
            },
            'DetectionThreshold': 0,
            'confidence_percentage': 0,
            'result': "",
            'error_message': "No Need to process The video",
            'video_path': vidpath,
            'heatmap': "",
            'processed_faces': "",
            'graph': ""
        }
            LOGGER.info("No Need to process The video") 
            results_details_json = json.dumps(results_details, indent=4).replace('\n', '').replace(' ', '')
            video_details_file_path = os.path.join(results_dir, 'results.json')
            with open(video_details_file_path, 'w') as f:
             f.write(results_details_json) 
            return video_details_file_path
        faces_t = torch.stack([transf(image=frame['faces'][0])['image']
                          for frame in vid_faces if len(frame['faces']) > 0]) 
    except Exception as e:  
        results_details = {
            'video_details': {
            'FPS': int(fps),
            'Resolution': resolution,
            'VideoCodec': video_codec,
            'DurationSeconds': int(duration_seconds),
            },
            'DetectionThreshold': 0,
            'confidence_percentage': 0,
            'result': "",
            'error_message': "Error Processing The video",
            'video_path': vidpath,
            'heatmap': "",
            'processed_faces': "",
            'graph': ""
        }
        logging.basicConfig(filename='test1.log',filemode='w',format='%(name)s - %(levelname)s - %(message)s')
        LOGGER.info("Error processing video") 
        results_details_json = json.dumps(results_details, indent=4).replace('\n', '').replace(' ', '')
        video_details_file_path = os.path.join(results_dir, 'results.json')
        with open(video_details_file_path, 'w') as f:
          f.write(results_details_json)  
        return video_details_file_path
    

    fps, resolution, video_codec, total_frames, duration_seconds, video_size_mb = get_media_details(vidpath)
    # Check if the video file size is more than 300 MB
    max_video_size_mb = 300
    if video_size_mb > max_video_size_mb:          
            # Construct the results_details dictionary with error information
            results_details = {
                'video_details': {
                    'FPS': int(fps),
                    'Resolution': resolution,
                    'VideoCodec': video_codec,
                    'DurationSeconds': int(duration_seconds),
                    'Size': f"{video_size_mb:.2f} MB"
                },
                'DetectionThreshold': 0,
                'confidence_percentage': 0,
                'result': "",
                'error_message': "Video file size more than 300 MB",
                'video_path': vidpath,
                'heatmap': "",
                'processed_faces': "",
                'graph': ""
            }
            LOGGER.info(f"{current_datetime} - Video file size is more than {max_video_size_mb} MB ({processing_time} seconds)")
            results_details_json = json.dumps(results_details, indent=4).replace('\n', '').replace(' ', '')
            video_details_file_path = os.path.join(results_dir, 'results.json')
            with open(video_details_file_path, 'w') as f:
                f.write(results_details_json)

            return video_details_file_path

    ## It will return the video processing result with confidence score and result real or fake    
    with torch.no_grad():
        faces_pred = net(faces_t.to(device)).cpu().numpy().flatten()   
    confidence_percentage = expit(faces_pred.mean())
    confidence_percentage = str(confidence_percentage)
    confidence_percentage = confidence_percentage
    confidence_percentage = float(confidence_percentage)
    confidence_percentage = str(confidence_percentage)
    confidence_percentage = confidence_percentage
    confidence_percentage = float(confidence_percentage)
    #logging.info("Confidence Percentage: {}".format(confidence_percentage))
    logging.info(f"{current_datetime} - Confidence Percentage: {confidence_percentage:.2f} ({processing_time} seconds)")
    if (confidence_percentage < threshold):

        result = "Real"
        #logging.info("Result: {}".format(result))
        logging.info(f"{current_datetime} - Result: {result}({processing_time} seconds)")
    else:
        result = "Fake"
        #logging.info("Result: {}".format(result))
        logging.info(f"{current_datetime} - Result: {result}({processing_time} seconds)")
    text = "Confidence Percentage: {:.2f}\nResult: {}".format(
        confidence_percentage, result)
## This function is basically use to Find out the heatmap images and form heatmap images and save it to gif format
    def generate_gradcam_patches(image, heat_map, threshold=0.5,cmap='viridis'):
       # Resize heat map to match the image size
        height, width = image.shape[:2]
        heat_map_resized = transform.resize(heat_map, (height, width))

        # Threshold the heat map to identify important regions
        thresholded_heat_map = (heat_map_resized > threshold).astype(np.uint8)

        # Find contours of the thresholded heat map
        contours, _ = cv2.findContours(thresholded_heat_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an image with the same shape as the input image
        gradcam_patches = np.zeros_like(image)

        # Draw filled contours on the gradcam_patches image
        for i, contour in enumerate(contours):
            # Use a different color for each contour based on the colormap
            color = plt.cm.get_cmap(cmap)(i / len(contours))[:3]
            color = tuple(int(255 * c) for c in color)
            cv2.drawContours(gradcam_patches, [contour], -1, color, thickness=cv2.FILLED)

        return gradcam_patches
    def heatmapadd(image, heat_map, alpha=0.6, display=False, save=None,axis='on',cmap='viridis', verbose=False):
        height, width = image.shape[:2]

        # Resize heat map to match the image size
        heat_map_resized = transform.resize(heat_map, (height, width))

        # Normalize heat map
        max_value = np.max(heat_map_resized)
        min_value = np.min(heat_map_resized)
        normalized_heat_map = (heat_map_resized - min_value) / (max_value - min_value)

        # Generate Grad-CAM-like patches
        gradcam_patches = generate_gradcam_patches(image, normalized_heat_map)

        # Display the original image and overlay the Grad-CAM-like patches
        plt.imshow(image)
        plt.imshow(gradcam_patches, alpha=alpha,cmap=cmap)
        plt.axis(axis)

        # Get the current figure as a skimage image
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = io.imread(buf)

        return img
    # Assuming you have a list of binary face masks (face_masks) corresponding to each face image
    faces_t = torch.stack([transf(image=frame['faces'][0])['image']
                        for frame in vid_faces if len(frame['faces']) > 0])

    with torch.no_grad():
        faces_pred = net(faces_t.to(device)).cpu().numpy().flatten()

    with torch.no_grad():
        if hasattr(net, 'feat_ext'):
            atts = net.feat_ext.get_attention(faces_t.to(device)).cpu()
        else:
            atts = net.get_attention(faces_t.to(device)).cpu()

    vid_faces_t = [frame['faces'][0] for frame in vid_faces if len(frame['faces'])]
    output_gif_file_name = f'{vidpath}_gradcam_patches.gif'
    gif_writer = imageio.get_writer(output_gif_file_name, mode='I', duration=0.5)

    for idx, (face_t, att) in enumerate(zip(vid_faces_t, atts)):
        face_im = ToPILImage()(face_t)
        att_img = ToPILImage()(att)
        face_im_skimage = np.array(face_im)
        att_img_skimage = np.array(att_img)
        im = heatmapadd(face_im_skimage, att_img_skimage, alpha=0.6, axis='off')
        gif_writer.append_data(im)

    gif_writer.close()
    #print(f"Output GIF file saved as: {output_gif_file_name}")
	## Here inside the framwise graph with result will be displayed
    with torch.no_grad():
        faces_pred = net(faces_t.to(device)).cpu().numpy().flatten()
        x_values = [f['frame_idx'] for f in vid_faces if len(f['faces'])]
        y_values = expit(faces_pred.flatten())
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.stem(x_values, y_values)
        ax.set_title('Fakeness Score Display')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Score')
        ax.set_ylim([0, 1])
        ax.grid(True)
        x=fig.savefig(os.path.join(results_dir, 'graph.png'), dpi=60)
    ## It will save the results such that vidpath,confidence_percentage,result,output_gif_file_name,graph inside dictionary format
      # Get the current date and time
     # Create a dictionary to store video details
    #video_src = os.path.join('video', vidpath)
    video_src=vidpath
    video_copy_path = os.path.join(results_dir, os.path.basename(vidpath))
      # Make a copy of the video file in the same directory
    #shutil.copy2(video_src, video_copy_path)
    logging.info(f"{current_datetime} - Source Video File Directory: {video_src} ({processing_time} seconds)") 
    dest_dir = os.path.join('results', timestamp)
    logging.info(f"{current_datetime} - Target Directory: {dest_dir} ({processing_time} seconds)") 
    Target_video_path =shutil.copy2(video_src, video_copy_path)
    # Copy the video file to the processing directory
    logging.info(f"{current_datetime} - Target Video File Directory: {Target_video_path} ({processing_time} seconds)") 
    gif_src = output_gif_file_name
    #gif_src=output_gif_file_name
    logging.info(f"{current_datetime} - Gif source File Directory: {gif_src} ({processing_time} seconds)")
    Target_gif_path= shutil.move(gif_src, dest_dir)
    logging.info(f"{current_datetime} - Target Gif File Directory: {Target_gif_path} ({processing_time} seconds)") 
    face_directory_src=os.path.join(faces_directory)
    logging.info(f"{current_datetime} - Processed Faces Source Directory: {face_directory_src} ({processing_time} seconds)") 
    Target_Processed_Faces=shutil.move(face_directory_src,dest_dir)
    logging.info(f"{current_datetime} - Processed Faces Target Directory: {Target_Processed_Faces} ({processing_time} seconds)") 
    graph_path = os.path.join(results_dir, 'graph.png')
    logging.info(f"{current_datetime} - Output Graph Directory: {graph_path} ({processing_time} seconds)") 
     # Make a copy of the video file in the same directory
     # Construct the results_details dictionary
    # Call get_media_details to get video details
    results_details = {
        'video_details': {
            'FPS': int(fps),
            'Resolution': resolution,
            'VideoCodec': video_codec,
            'DurationSeconds': int(duration_seconds),
        },
        'DetectionThreshold': threshold,
        'confidence_percentage': confidence_percentage,
        'result': result,
        'video_path': Target_video_path,
        'heatmap': Target_gif_path,
        'processed_faces': Target_Processed_Faces,
        'graph': graph_path
    }
    # Convert the dictionary to JSON format
    results_details_json = json.dumps(results_details, indent=4).replace('\n', '').replace(' ', '')
    video_details_file_path = os.path.join(results_dir, 'results.json')
    with open(video_details_file_path, 'w') as f:
        f.write(results_details_json)
       # Log the video details to the logger
    logging.info(f"{current_datetime} - Video details: {results_details_json} ({processing_time} seconds)")   
    logging.info(f"{current_datetime} - Results File Directory: {results_details_json} ({processing_time} seconds)") 
    ## It will return the result_list here
    return video_details_file_path
if __name__ == "__main__":
      # Parse command-line arguments to get the video path
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_file', type=str,nargs='+')
    args = parser.parse_args()
    video_path = ' '.join(args.in_file)
    # Define a regular expression pattern to match the video path
    # Call the core_processing_unit function with the video path and get the result
    prediction_result = core_processing_unit(video_path)
    # Print or use the prediction_result as needed
    print(f"prediction_result : \n {prediction_result}")
    
   

    
   
