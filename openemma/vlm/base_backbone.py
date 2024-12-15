import cv2
import os
import re
import torch
import json
import numpy as np

from PIL import Image
from utils import query_gpt4
from transformers import MllamaForConditionalGeneration, AutoProcessor
from openemma.visualize.visualize import CamParams
from openemma.YOLO3D.inference import yolo3d_nuScenes
from scipy.spatial.transform import Rotation as R


class BaseOpenEMMA:
    def __init__(self, args):
        super(BaseOpenEMMA, self).__init__()
        self.model_path = args.model_id
        self.init_model(args=args)

    def init_model(self, args=None):
        if "Llama-3.2" in self.model_path:
            self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
            self.processor = AutoProcessor.from_pretrained(self.model_path)

        elif "gpt" in self.model_path:
            self.api_key = args.api_key
            
        
    def getMessage(self, prompt):
        message = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
        return message

    def vlm_inference(self, text=None, image_path=None, sys_message=None, args=None):
        if  "Llama-3.2" in self.model_path:
            image = Image.open(image_path).convert('RGB')
            message = self.getMessage(text)
            input_text = self.processor.apply_chat_template(message, add_generation_prompt=True)
            inputs = self.processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(self.model.device)

            output = self.model.generate(**inputs, max_new_tokens=512)

            output_text = self.processor.decode(output[0])

            if "Llama-3.2" in self.model_path:
                output_text = re.findall(r'<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>', output_text, re.DOTALL)[0].strip()
            return output_text
        elif "gpt" in self.model_path:
            output_text = query_gpt4(
            text, api_key=self.api_key, 
            image_path=image_path,
            sys_message=sys_message
            )
            return output_text
        
    def getCoT(self, image_path, ego_fut_diff, ego_fut_trajs, ego_his_diff, backbone):
        r1 = self.Scenedescription(image_path, backbone)
        r2 = self.get_objects(image_path)
        r3 = self.description_criticalobjects(r2, image_path, backbone)
        r4 = self.Metadecision(ego_fut_diff, ego_fut_trajs, ego_his_diff)
        rationale = f"""Scene description:\n{r1}\nCritial objects:\n{r2}\nBehavior description:\n{r3}\nMeta driving decision:\n{r4}"""
        return rationale

    def Scenedescription(self,image_path, backbone):
        prompt = "Provide a short description of driving scenarios."
        r1=self.vlm_inference(text=prompt, image_path=image_path)

        return r1
    
    def description_criticalobjects(self,r2, image_path, backbone):
        prompt=f"Identified critical object: {r2}\nProvide a short description of the current status and intended actions to the above identified critical object for the ego car."
        r3 = self.vlm_inference(text=prompt, image_path=image_path)
        # r3 = backbone.vlm_inference(prompt, image_path)
        return r3

    def Metadecision(self, ego_fut_diff, ego_fut_trajs, ego_his_diff):
        r4 = self.compute_meta_action(ego_fut_diff, ego_fut_trajs, ego_his_diff)
        return r4

    def compute_meta_action(self, ego_fut_diff, ego_fut_trajs, ego_his_diff):
        meta_action = ""

        constant_eps = 0.5
        his_velos = np.linalg.norm(ego_his_diff, axis=1)
        fut_velos = np.linalg.norm(ego_fut_diff, axis=1)
        cur_velo = his_velos[-1]
        end_velo = fut_velos[-1]

        if cur_velo < constant_eps and end_velo < constant_eps:
            return "STOPPED"
        elif end_velo < constant_eps:
            speed_meta = "DECELERATING TO STOP"
        elif np.abs(end_velo - cur_velo) < constant_eps:
            speed_meta = "CONSTANT SPEED FORWARD"
        else:
            if cur_velo > end_velo:
                if cur_velo > 2 * end_velo:
                    speed_meta = "QUICKLY DECELERATING"
                else:
                    speed_meta = "DECELERATING"
            else:
                if end_velo > 2 * cur_velo:
                    speed_meta = "QUICKLY ACCELERATING"
                else:
                    speed_meta = "ACCELERATING"
        
        forward_th = 2.0
        lane_changing_th = 4.0

        if (np.abs(np.array(ego_fut_trajs)[:, 0]) < forward_th).all():
            behavior_meta = "MOVING FORWARD"
        else:
            if np.array(ego_fut_trajs)[-1, 0] < 0:  # left
                if np.abs(ego_fut_trajs[-1, 0]) > lane_changing_th:
                    behavior_meta = "TURNING LEFT"
                else:
                    behavior_meta = "LANE CHANGING LEFT"
            elif np.array(ego_fut_trajs)[-1, 0] > 0:  # right
                if np.abs(np.array(ego_fut_trajs)[-1, 0]) > lane_changing_th:
                    behavior_meta = "TURNING RIGHT"
                else:
                    behavior_meta = "LANE CHANGING RIGHT"
            else:
                raise ValueError(f"Undefined behaviors: {ego_fut_trajs}")

        meta_action = speed_meta + " AND " + behavior_meta
        return meta_action

    def compute_command(self, ego_fut_trajs):
        lane_changing_th = 4.0

        if (np.abs(np.array(ego_fut_trajs)[:, 0]) < lane_changing_th).all():
            return "MOVE FORWARD"  
        elif np.array(ego_fut_trajs)[-1, 0] < 0:
            return "TURN LEFT"
        elif np.array(ego_fut_trajs)[-1, 0] > 0:
            return "TURN RIGHT"
        else:
            raise ValueError(f"Undefined behaviors: {ego_fut_trajs}")


    def generate_waypoints(self, command, image_path, data=None, backbone=None, args=None):
        ego_fut_diff = data["gt_ego_fut_diff"]
        ego_fut_trajs = data["gt_ego_fut_trajs"]
        ego_his_diff = data["gt_ego_his_diff"]
        ego_his_trajs = str(data["gt_ego_his_trajs"]).replace("\n", '')

        rationale = self.getCoT(image_path, ego_fut_diff, ego_fut_trajs, ego_his_diff, backbone)
        sys_message = "You have access to a surround-view camera image of the ego vehicle, a high-level intent command, a sequence of past ego waypoints, and a driving rationale. Each waypoint is represented as [x, y], where x corresponds to the lateral (left-right) position, and y corresponds to the longitudinal (front-back) position. Your task is to predict future waypoints for the ego vehicle over the next 10 seconds, ensuring the y-coordinate reaches approximately 10. Make sure it is a smooth waypoint\n\n"
        prompt = f"""##High-level command:{command}\n\n##Historical waypoints of the ego car:{ego_his_trajs}\n\n##Driving rationale:\n{rationale}\n\nOnly generate the predicted future waypoints only in the format [x1,y1], [x2,y2],...,[xn,yn]. As the ego vehicle progresses forward, the y-coordinates of future waypoints should remain positive and exhibit a steadily increasing trend over time, reflecting forward movement. Make sure the y-coordinate of the last predicted waypoint reaches approximately 10.\n\n##Future waypoints:"""
        output_text = self.vlm_inference(text=prompt, image_path=image_path, sys_message=sys_message)
        return output_text

    def spatial_reasoning(self, image_path):
        prompt= "Detect all objects in 3D space and generate their 3D bounding boxes in the following format: [x, y, z, l, w, h, theta, cls]. Here:\n[x, y, z] specifies the object's center location in the vehicle frame,\n[l,w,h] represent the length, width, and height of the bounding box,\ntheta is the object's heading angle, and\n cls is the class label as a text identifier.\nPlease ensure that all objects are accurately identified, with precise bounding boxes according to the specified parameters."

        output_text = self.vlm_inference(text=prompt, image_path=image_path)
        
        return output_text

    def road_graph(self, image_path):
        prompt = "Predict the drivable lanes in the provided driving scene by generating an ordered sequence of waypoints in the format x1,y1;x2,y2;...;xn,yn. Each waypoint x,y should be a floating-point number with a precision of two decimal places. Ensure the sequence accurately reflects the drivable lane paths for a precise and smooth lane trajectory."

        output_text = self.vlm_inference(text=prompt, image_path=image_path)
        return output_text

    def blockage_detect(self, bbx, image_path):
        prompt = "Here is a list of objects currently in front of the ego car. Based on this information, determine if the road ahead is temporarily blocked.\n" + bbx

        output_text = self.vlm_inference(text=prompt, image_path=image_path)
        return output_text

    def get_objects(self, image_path):
        prompt = "Plesase list 2-3 key objects the ego car should focus on, specifying only the object's name and its location within the image of the driving scene. Only output the name of the object and its related location in the image of the driving scence without any other information."
        output_text = self.vlm_inference(text=prompt, image_path=image_path)
        return output_text

    def fix_traj(self, traj):
        if np.abs(traj[0][0]) > 0.0:
            for i in range(len(traj)):
                if traj[i][0] > 0.0:
                    traj[i][0] -= np.abs(traj[0][0])
                else:
                    traj[i][0] += np.abs(traj[0][0])
        return traj

    def frames_to_video(self, input_folder, output_file, fps):
        images = [img for img in os.listdir(input_folder) if img.endswith(".jpg") or img.endswith(".png")]
        images.sort()  # Sort files by name to maintain order

        frame = cv2.imread(os.path.join(input_folder, images[0]))
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        for image in images:
            img_path = os.path.join(input_folder, image)
            frame = cv2.imread(img_path)
            out.write(frame) 
        
        out.release()
        print(f"Video saved as {output_file}")

    def plot_bbx_yolo(self, input_folder, output_file, raw_file, roi_r=25.0, roi_w=5.0, roi_d=100.0):
        images = [img for img in os.listdir(input_folder) if img.endswith(".jpg") or img.endswith(".png")]
        images.sort()  

        with open(raw_file, 'r') as file:
            raw_data = json.load(file)
        
        for i in range(len(images)):
            image = images[i]
            img_path = os.path.join(input_folder, image)
            yolo3d_nuScenes(img_path, output_file, roi_r=-1, roi_w=-1, roi_d=-1)

