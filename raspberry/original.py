import time
from picamera2 import Picamera2

import torch
import numpy as np
from torchvision import models, transforms

import cv2
from PIL import Image
import timm
from transformers import MobileViTForImageClassification
torch.backends.quantized.engine = "qnnpack"


# Initialize the camera, configure it and start it
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (256, 256)}))
picam2.start()


classes = {
  0: "Bacterial_spot",
  1: "Early_blight",
  2: "Late_blight",
  3: "Leaf_Mold",
  4: "Septoria_leaf_spot",
  5: "Curl_Virus",
  6: "Tomato_mosaic_virus",
  7: "Healthy"
}

preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

total_time_inf = 0

# note that for mobilevit-small, is necessary to use the MobileViTForImageClassification class
# net = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")

path = 'model.pt'
net = torch.jit.load(f"{path}", map_location=torch.device("cpu"))
#or
net = torch.load(f"{path}", map_location=torch.device("cpu"))

device = torch.device("cpu")
net.to(device)
net.eval()


start_time = time.time()
last_logged = time.time()
frame_count = 0
max_frame=100

# Calculate FPS
with torch.no_grad():
    while frame_count<max_frame:

        # capture a frame
        frame = picam2.capture_array()

        
        if frame is None:
            raise RuntimeError("failed to read frame")

        frame = cv2.resize(frame, (224, 224))
        
        # display the frame
        cv2.imshow('Camera', frame)
        
        # convert opencv output from BGR to RGB
        frame = frame[:, :, [2, 1, 0]]
        permuted = frame

        # preprocess the frame
        input_tensor = preprocess(frame).to(device)

        # create a mini-batch as expected by the model
        input_batch = input_tensor.unsqueeze(0)

        # run model
        start_time_i = time.time()
        output = net(input_batch)
        end_time_i = time.time()
        now = time.time()

        total_time_inf += (end_time_i-start_time_i)*1000
		
        frame_count += 1
        

        top = list(enumerate(output[0].softmax(dim=0)))
        top.sort(key=lambda x: x[1], reverse=True)
        
        # uncomment for debugging
        #for idx, val in top[:1]:
            #print(f"{val.item()*100:.2f}% {classes[idx]}" + f"- {frame_count}")

        # exit if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break

end_time = time.time()


# Calculate FPS and Inference time in ms
total_time = end_time - start_time
tot_inf = total_time_inf / max_frame
average_time_per_iteration = total_time / max_frame
fps = 1.0 / average_time_per_iteration
print(f"{fps} FPS - inference time: {tot_inf}" )           

cv2.destroyAllWindows()
