import cv2
import os
import glob
import pdb

# Parameters for the video
output_video = 'eventssubsequently.mp4'  # Output video file name
fps = 30  # Frames per second
frame_size = (256, 256)  # Width, height of the images (must match your image size)

# Create a VideoWriter object (you can change the codec to suit your platform)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec (XVID for .avi or MP4V for .mp4)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec (XVID for .avi or MP4V for .mp4)
video_writer = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

images_folder = '/home/ziyan/02_research/EventHPE/eventsOnImgs'
image_files = sorted(glob.glob(os.path.join(str(images_folder), "*.jpg")))

# Assuming `images` is a list of file paths or images
for i in range(len(image_files)):  # Loop over the images
    # Read the current image (make sure the image size matches the frame size)
    imageCurr = cv2.imread(image_files[i])  # Load image

    # Optional: Resize image if needed (it must match the frame_size)
    imageCurr = cv2.resize(imageCurr, frame_size)

    # Write the frame to the video
    video_writer.write(imageCurr)

# Release the VideoWriter
video_writer.release()

# If you want to view the generated video immediately:
cv2.destroyAllWindows()
