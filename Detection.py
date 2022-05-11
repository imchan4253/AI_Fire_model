
import cv2

import numpy as np

import onnxruntime

import math

from motpy import Detection, MultiObjectTracker
from pprint import pprint

import glob
import time

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

    model_name = "ppp_hub_b.onnx"
    aaa = glob.glob('/workspace/sam_person/_aibox/new_vvideo/guro_fire.*')
    for filePath in aaa:
        with open(filePath, 'rb') as f:
             dataset=f.read()
             path = str(filePath.split('/')[5])
             PPath = "/workspace/sam_person/_aibox/imchanyoung_out/"+str(path.split('.')[0])+".txt"




             #record = "/workspace/sam_person/_aibox/imchanyoung_out/"+str(path.split('.')[0])+"("+str(model_name)+")"+".mp4"
             record = "/workspace/sam_person/_aibox/imchanyoung_out/"+str(path.split('.')[0])+" 0.25.mp4"

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

             writer = cv2.VideoWriter(record, fourcc, fps=40, frameSize=(WWW, HHH))



             pproc = PostProcessing()


             frame_idx = 0
             Detection_count = 0
             count = {}
             countt = {}
             tracker = MultiObjectTracker(dt=0.1, tracker_kwargs={'max_staleness': 10}, model_spec='constant_acceleration_and_static_box_size_2d', matching_fn_kwargs={'min_iou': 0.25})
             #tracker = MultiObjectTracker(dt=0.1, model_spec='constant_acceleration_and_static_box_size_2d', matching_fn_kwargs={'min_iou': 0.25})
             #detections = []
             while True:

                 ret, img = capture.read()

                 frame_idx += 1



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



                 #print(f"input image shape: {img.shape}")
                 output = sess.run(["output"], {input_name: img})[0]
                 boxes = nms_bbox(output[0][:,:4])
                 classes = output[0][:, 4:5]
                 choosed_idx = np.max(classes, axis=-1) > 0.48

                 valid_classes = classes[choosed_idx]
                 valid_boxes = boxes[choosed_idx]


                 confidences = np.max(valid_classes, axis=-1)
                 ids = np.argmax(valid_classes, axis=-1)


                 filtered_boxes = pproc.bbox_filter(confidences, ids, valid_boxes, 0.45)
                 pproc.update_tracker(filtered_boxes)
                 county = 0
                #  detections = []
                 #detections.append(Detection(box=object_box))
                 #tracks  = tracker.step(detections=detections)
                 if 0 in filtered_boxes:
                    fire_shape = filtered_boxes[0].shape
                    fire_count = fire_shape[0]
                    #print("fires detected : " + str(fire_count))
                    #detections = []

                    def convert_box_to_detection(fire_box):
                        (conf,X, Y, W, H) = fire_box
                        rate = round(conf,2)
                        x0 = float((X-W/2)*WWW)
                        y0 = float((Y-H/2)*HHH)
                        x1 = float((X+W/2)*WWW)
                        y1 = float((Y+H/2)*HHH)

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

                        object_box = np.array([X0, Y0, X1, Y1])
                        return Detection(box=object_box)

                    detections = list(map(convert_box_to_detection, filtered_boxes[0][:, 0:]))
                    tracks = tracker.step(detections=detections)

                    for track in tracks:
                        XX0 = int(track.box[0])
                        YY0 = int(track.box[1])
                        XX1 = int(track.box[2])
                        YY1 = int(track.box[3])
                        ABA = (XX1-XX0)*(YY1-YY0)
                        CVC = WWW*HHH
                        ID = track.id[:8]
                        RATIO= round(ABA/CVC, 4)
                        #VIDEO_time = int(frame_idx/30)
                        #cv2.rectangle(original, (int(track.box[0]), int(track.box[1])), (int(track.box[2]), int(track.box[3])), (0, 255, 0), 1)
                        #print(VIDEO_time,"sssssssss")
                        #print(track,"aaaaaaaaa")
                        if track.id in count.keys():
                            count[track.id] += 1
                            #countt[VIDEO_time] += 1
                        else:
                            count[track.id] = 1
                            #countt[VIDEO_time] = 1
                        #print(countt[1])
                        if count[track.id] > 0 :
                           #VIDEO_time = int(frame_idx/30) 
                            cv2.rectangle(original, (int(track.box[0]), int(track.box[1])), (int(track.box[2]), int(track.box[3])), (0, 255, 0), 1)
                            cv2.putText(original, "fire", (int(track.box[0]+5), int(track.box[1]+40)), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0],2)
                            print("fire", (int(track.box[0]), int(track.box[1])), (int(track.box[2]), int(track.box[3])))
                            #print(str(path.split('.')[0]),ID,str(count[track.id]),RATIO)
                           #if VIDEO_time in countt.keys():
                               #countt[VIDEO_time]+=1
                           #else:
                               #countt[VIDEO_time]=1
                           #print(countt)


                    # for fire_box in filtered_boxes[0][:, 0:]:
                    #     (conf,X, Y, W, H) = fire_box
                    #     rate = round(conf,2)
                    #     x0 = float((X-W/2)*WWW)
                    #     y0 = float((Y-H/2)*HHH)
                    #     x1 = float((X+W/2)*WWW)
                    #     y1 = float((Y+H/2)*HHH)      ###########Sclase coords##########3

                    #     gain = min(img.shape[2:][0] / original.shape[0], img.shape[2:][1] / original.shape[1])  # gain  = old / new
                    #     pad = (img.shape[2:][1] - original.shape[1] * gain) / 2, (img.shape[2:][0] - original.shape[0] * gain) / 2 
                    #     X0 = x0-pad[0]
                    #     Y0 = y0-pad[1]
                    #     X1 = x1-pad[0]
                    #     Y1 = y1-pad[1]
                    #     X0 = round(X0/gain)
                    #     Y0 = round(Y0/gain)
                    #     X1 = round(X1/gain)
                    #     Y1 = round(Y1/gain)
                    #     X0 = int(X0)
                    #     Y0 = int(Y0)
                    #     X1 = int(X1)
                    #     Y1 = int(Y1)
                    #     #detections = []
                    #     object_box = np.array([X0, Y0, X1, Y1])
                    #     detections.append(Detection(box=object_box))


                        # tracks  = tracker.step(detections=detections)
                    #print(tracks, "ssssssssssssssss")
                        # id_counts = 0
                    #for track in tracks:
                 #detections = []
                        # for track in tracks:
                        #for (j,box) in enumerate(track.box):
                            # XX0 = int(track.box[0])
                            # YY0 = int(track.box[1])
                            # XX1 = int(track.box[2])
                            # YY1 = int(track.box[3])
                            # ABA = (XX1-XX0)*(YY1-YY0)
                            # CVC = WWW*HHH
                            # ID = track.id[:8]
                            # RATIO= round(ABA/CVC, 4)
                            # VIDEO_time = round(frame_idx/30)
                            # id_counts += 1
                            # print(track)
                            # try : count[track.id] += 1
                            # except : count[track.id] = 1
                            # AAA = list(count.values())
                            # #print(AAA)
                            # print(count[track.id])
                            # try :
                            #    i = -1
                            #    while True :
                            #        i += 1
                            #        print(AAA[i])


                            # except :
                            #    if AAA[i-1] > 0  :
                            #       cv2.rectangle(original, (int(track.box[0]), int(track.box[1])), (int(track.box[2]), int(track.box[3])), (0, 255, 0), 1)
                            #       cv2.putText(original, "fire", (int(track.box[0]+5), int(track.box[1]+40)), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0],2)
                            #       print("fire", (int(track.box[0]), int(track.box[1])), (int(track.box[2]), int(track.box[3])))
                            #       print(RATIO)

                            # if count[track.id] > 17:
                            #     cv2.rectangle(original, (int(track.box[0]), int(track.box[1])), (int(track.box[2]), int(track.box[3])), (0, 255, 0), 1)
                            #     cv2.putText(original, "fire", (int(track.box[0]+5), int(track.box[1]+40)), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0],2)
                            #     print("fire", (int(track.box[0]), int(track.box[1])), (int(track.box[2]), int(track.box[3])))
                            #     print(RATIO)
 

                 writer.write(original)
             writer.release()





