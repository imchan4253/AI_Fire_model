import cv2

import numpy as np

import onnxruntime

import math

from motpy import Detection, MultiObjectTracker
#import torch
#torch.ops.torchvision.nms
#import torch.backends.cudnn as cudnn
from pprint import pprint

import glob
import time
#import torchvision

class PostProcessing():

    def __init__(self):

        self.tracker = MultiObjectTracker(dt=0.1)





    def bbox_filter(self, confs, ids, boxes, thres=0.4):

        confs = np.expand_dims(confs, axis=-1)

        ids = np.expand_dims(ids, axis=-1)



        group = {}

        features = np.concatenate([ids, confs, boxes], axis=-1)



        for f in features:

            key = f[0].astype(np.int32)

            new_conf_box = np.expand_dims(f[1:], axis=0)

            if key in group:

                conf_box = group[key]

                new_conf_box = np.concatenate((conf_box, new_conf_box))



            group[key] = new_conf_box





        new_group = {}

        for key in group:

            mat = group[key]

            indices = []

            for r, vec in enumerate(mat):

                ious = self._get_iou(vec, mat) 

                c = np.where(ious > thres)[0]

                high_conf = np.argmax(mat[c][:,0])

                if r == c[high_conf]:

                    indices.append(r)



            new_group[key] = mat[indices]



        return new_group





    def parse_bboxes(self, bboxes):

        for objId in bboxes:

            conf = bboxes[objId][:, 0]



            tuple_int = lambda t: tuple(map(int, t))

            for box in bboxes[objId][:, 1:]*448:

                x, y, w, h = tuple(box)


                yield tuple_int((x-w/2,y-h/2,x+w/2,y+h/2))





    def update_tracker(self, result):

        detections = list(map(lambda coord: Detection(box=np.array(coord)), self.parse_bboxes(result)))

        self.tracker.step(detections=detections)





    def get_tracks(self):

        tracks = self.tracker.active_tracks()

        for t in tracks:

            yield tuple(map(int, t.box))





    def _get_iou(self, vec, mat):

        conf, x, y, w, h = tuple(vec)

        confs, xs, ys, ws, hs = mat[:, 0], mat[:, 1], mat[:,2], mat[:, 3], mat[:,4]



        intersection = (w/2 + ws/2 - np.abs(x - xs)) * (h/2 + hs/2 - np.abs(y - ys))

        union = w * h + ws * hs - intersection



        return intersection/union 



def nms_bbox(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x
    y[:, 0] = x[:, 0] / WWW  # top left x
    y[:, 1] = x[:, 1] / HHH  # top left y
    y[:, 2] = x[:, 2] / WWW  # bottom right x
    y[:, 3] = x[:, 3] / HHH  # bottom right y
    return y



def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, auto_size=32):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, auto_size), np.mod(dh, auto_size)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


if __name__ == "__main__":

    model_name = "best_smoke.onnx"
    #model_name = ".onnx"

    #path ="b_10m_on.avi"
    aaa = glob.glob('/workspace/sam_person/_aibox/r_smoke/SMOKE.*')
    for filePath in aaa:
        with open(filePath, 'rb') as f:
             dataset=f.read()
             path = str(filePath.split('/')[5])
             PPath = "/workspace/sam_person/_aibox/yolot/"+str(path.split('.')[0])+".txt"




             record = "/workspace/sam_person/_aibox/yolor_smoke_out/"+str(path.split('.')[0])+"_R_out.mp4"

             capture = cv2.VideoCapture(filePath)
             print(int(capture.get(3)))
             WWW = int(capture.get(3))
             print(int(capture.get(4)))
             HHH = int(capture.get(4))


             sess_options = onnxruntime.SessionOptions()

             sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

             #sess = onnxruntime.InferenceSession(model_name, sess_options=sess_options)

             sess = onnxruntime.InferenceSession(model_name, None)
             sess.set_providers(['CUDAExecutionProvider'])



             input_name = sess.get_inputs()[0].name
             #print(sess.get_inputs()[0].name)
             output0_name = sess.get_outputs()[0].name
             #print(sess.get_output().name)
             #print(sess.get_output()[0].name)
             #print(sess.get_output()[1].name)
             #output1_name = sess.get_outputs()[1].name
             #print(sess.get_outputs().name)
             print("onnxruntime session is ready to run")




            





             fourcc = cv2.VideoWriter_fourcc(*"mp4v")

             #writer = cv2.VideoWriter(record, fourcc, fps=30)
             writer = cv2.VideoWriter(record, fourcc, fps=30, frameSize=(WWW, HHH))



             pproc = PostProcessing()
             #pproc1 = PostProcessing()


             frame_idx = 0

             while True:

                 ret, img = capture.read()

                 frame_idx += 1



        # for demo recording

        #if frame_idx > 200

            #ret = False



                 if not ret:

                     print("invalid video format")

                     writer.release()

                     break


                 original = img


                 img = letterbox(img, new_shape=640,auto_size=64)[0]
                 img = img.astype(np.float32)

                 img /= 255

                 img = np.transpose(img, [2, 0, 1])

                 img = np.array((img[2], img[1], img[0]), dtype=np.float32)

                 img = np.expand_dims(img, axis=0)


                #return path, img, img0, self.cap 
                 #print(img)

                 print(f"input image shape: {img.shape}")
               
        
                 output = sess.run(["output"], {input_name: img})[0]
                 #boxes = xywh2xyxy(output[0][:,:4])
                 boxes = nms_bbox(output[0][:,:4])
                 #boxes = output[0][:,:4]
                 #print("ssssssssssssss",boxes)
                 #classes = output[0][:, 5:]*output[0][:, 4:5]
                 #class1 = output[0][:, 5:]
                 classes = output[0][:, 4:5]
                 #print("ksksksksksk",class1)
                 choosed_idx = np.max(classes, axis=-1) > 0.25

                 #print("1111111111",choosed_idx)




                 valid_classes = classes[choosed_idx]
                 #print("222222222",valid_classes)
                 valid_boxes = boxes[choosed_idx]
                 ##print("333333333",valid_boxes)


                 #valid_classes1 = classes1[choosed_idx1]

                 #valid_boxes1 = boxes1[choosed_idx1]




                 confidences = np.max(valid_classes, axis=-1)
                 #print("44444444",confidences)
                 ids = np.argmax(valid_classes, axis=-1)
                 #print("555555555",ids)


                 #co
                 #ids1 = np.argmax(valid_classes1, axis=-1)


                 filtered_boxes = pproc.bbox_filter(confidences, ids, valid_boxes, 0.45)
                 pproc.update_tracker(filtered_boxes)
                 #pproc1.update_tracker(filtered_boxes1)
                 #print(filtered_boxes1)
                 if 0 in filtered_boxes:
                    fire_shape = filtered_boxes[0].shape
                    #print(confidences[0])
                    fire_count = fire_shape[0]
                    #print(pproc.bbox_filter)
                    #rate = np.rint(100 * confidences)
                    #print(rate)
                    print("smoke detected : " + str(fire_count))
                    #cv2.putText(org, "fires detected : " + str(fire_count), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2, lineType=cv2.LINE_AA)
                    #cv2.putText(org, "fires detected : " + str(fire_count), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,255,0], 2, lineType=cv2.LINE_AA)

                    for fire_box in filtered_boxes[0][:, 0:]:
                        #print("zkzkzkzqkqkqkqkqk",fire_box)
                        #print(confidences[0])
                        
                        #(X, Y, W, H) = tuple(fire_box)
                        (conf,X, Y, W, H) = fire_box
                        #print("zkzkzkzqkqkqkqkqk",X)
                        rate = round(conf,2)
                        x0 = float((X-W/2)*WWW)
                        #print("x00000000", x0)
                        #X0 = int((X-W/2)*
                        y0 = float((Y-H/2)*HHH)
                        #print("x00000000", x0,y0)
                        x1 = float((X+W/2)*WWW)
                        y1 = float((Y+H/2)*HHH)      ###########Sclase coords##########3

                        gain = min(img.shape[2:][0] / original.shape[0], img.shape[2:][1] / original.shape[1])  # gain  = old / new
                        pad = (img.shape[2:][1] - original.shape[1] * gain) / 2, (img.shape[2:][0] - original.shape[0] * gain) / 2 
                        X0 = x0-pad[0]
                        Y0 = y0-pad[1]
                        X1 = x1-pad[0]
                        Y1 = y1-pad[1]
                        X0 = round(X0/gain)
                        Y0 = round(Y0/gain)
                        X1 = round(X1/gain)
                        Y1 = round(Y1/gain)
                        X0 = int(X0)
                        Y0 = int(Y0)
                        X1 = int(X1)
                        Y1 = int(Y1)


                        #det[:, :4] = scale_coords(img.shape[2:], det[0], original.shape).round()
                        #print(det[:, :4])
                        #cv2.putText(original, "smoke : "+str(rate)+"%", (X0+3,Y0-4), cv2.FONT_HERSHEY_SIMPLEX,0.5, 2, lineType=cv2.LINE_AA)
                        #cv2.rectangle(original, (X0,Y0), (X1,Y1), (0, 0, 255), 2)
                        cv2.rectangle(original, (X0,Y0), (X1,Y1), [0, 255, 0], 1)
                        #cv2.rectangle(original,(X0, Y0),(X1 +3, Y1-4),[0,255,0], -1)
                        #cv2.putText(original, "smoke : "+str(rate)+"%", (X0+5,Y0+40), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0],2)
                        cv2.putText(original, "smoke : "+str(rate), (X0+5,Y0+40), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0],2)

                        #cv2.rectangle(original, (x0,y0), (x1,y1), (0, 0, 255), 4)


                 writer.write(original)
             writer.release()

