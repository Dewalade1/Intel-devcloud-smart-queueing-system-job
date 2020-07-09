
import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys


class Queue:
    '''
    Class for dealing with queues
    '''
    def __init__(self):
        self.queues=[]

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max=q
            frame=image[y_min:y_max, x_min:x_max]
            yield frame
    
    def check_coords(self, coords):
        d={k+1:0 for k in range(len(self.queues))}
        for coord in coords:
            for i, q in enumerate(self.queues):
                if coord[0]>q[0] and coord[2]<q[2]:
                    d[i+1]+=1
        return d


class PersonDetect:
    '''
    Class for the Person Detection Model.
    '''

    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold
        self._ie_core = IECore()

        try:
            try:
                self.model=self._ie_core.read_network(model=self.model_structure, weights=self.model_weights)
            except AttributeError:
                self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        '''
        TODO: This method needs to be completed by you
    
        Load the model given IR files.
        Asynchronous requests made within.
        '''
        
        self.exec_network = self._ie_core.load_network(network=self.model, device_name=self.device,num_requests = 1)
        print('[script] Model loaded to IECore')
        
        return
        
    def predict(self, image):
        '''
        TODO: This method needs to be completed by you
        '''
        input_name=self.input_name
        input_image=self.preprocess_input(image)
        input_dict={input_name: input_image}
        print('[script] Created input dictionary')
        
        self.exec_network.start_async(request_id=0, inputs=input_dict)
        print('[script] Completed Asyncronous Inference on inputs')
        
        if self.exec_network.requests[0].wait(-1)==0:
            get_output=self.exec_network.requests[0].outputs[self.output_name]
            print('[script] Retrieved Inference output')
            
            coords=self.preprocess_outputs(get_output)
            output_coords, out_image=self.draw_outputs(coords, image)
        
        print('[script] Output image(s) have been processed')
        print('[script] Completed Inference')
        
        return output_coords, out_image
        
    
    def draw_outputs(self, coords, image):
        '''
        TODO: This method needs to be completed by you
        
        Draw bounding boxes to the frame
        
        Params
        frame: frame from camera/video
        result: list contains the result of inference
        
        return
        frame: frame with bounding box drawn on it
        '''
        
        output_coords=[]
        initial_h, initial_w, channel=image.shape
        
        for xmin, ymin, xmax, ymax in coords:
            x_min = int(xmin*initial_w)
            y_min = int(ymin*initial_h)
            x_max = int(xmax*initial_w)
            y_max = int(ymax*initial_h)
            
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0,0,0), 2)
            output_coords.append([x_min, y_min, x_max, y_max])
        
        print("[script] Bounding boxes have been drawn on all input frames")
            
        return output_coords, image
                

    def preprocess_outputs(self, outputs):
        '''
        TODO: This method needs to be completed by you
        '''
        
        coords=[]
        for box in outputs[0][0]:
            if box[2]>self.threshold:
                coords.append(box[3:])
                
        print("[script] Bounding box coordinates have been extracted from the output")
        
        return coords

    def preprocess_input(self, image):
        '''
        TODO: This method needs to be completed by you
        '''
        p_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_image = p_image.transpose((2,0,1))
        p_image = p_image.reshape(1,*p_image.shape)
        print('\n[script] Input image(s) are ready for inference')
        
        return p_image


def main(args):
    model=args.model
    device=args.device
    video_file=args.video
    max_people=args.max_people
    threshold=args.threshold
    output_path=args.output_path

    start_model_load_time=time.time()
    pd= PersonDetect(model, device, threshold)
    pd.load_model()
    total_model_load_time = time.time() - start_model_load_time

    queue=Queue()
    
    try:
        queue_param=np.load(args.queue_param)
        for q in queue_param:
            queue.add_queue(q)
    except:
        print("[script] error loading queue param file")

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("[script] Cannot locate video file: "+ video_file)
    except Exception as e:
        print("[script] Something else went wrong with the video file: ", e)
    
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)
    
    counter=0
    start_inference_time=time.time()

    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            counter+=1
            
            coords, image= pd.predict(frame)
            num_people= queue.check_coords(coords)
            print(f"[Result] Total People in frame = {len(coords)}")
            print(f"[Result] Number of people in queue = {num_people}")
            out_text=""
            y_pixel=25
            
            for k, v in num_people.items():
                out_text += f"No. of People in Queue {k} is {v} "
                if v >= int(max_people):
                    out_text += f" Queue full; Please move to next Queue "
                cv2.putText(image, out_text, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                out_text=""
                y_pixel+=40
            out_video.write(image)
            
        total_time=time.time()-start_inference_time
        total_inference_time=round(total_time, 1)
        fps=counter/total_inference_time

        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time)+'\n')
            f.write(str(fps)+'\n')
            f.write(str(total_model_load_time)+'\n')

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)
    
    args=parser.parse_args()

    main(args)