import os
import schedule
import redis
from services.HelmetModelService import HelmetModelService
from services.ClothesModelService import ClothesModelService
from utils.area import AreaUtils
from utils.object import ObjectUtils
from utils.utils import Utils
from utils.firebase import Firebase
from tools import generate_detections as gdet
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort import preprocessing, nn_matching
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from core.config import cfg
from tensorflow.python.saved_model import tag_constants
from core.yolov4 import filter_boxes
import core.utils as utils
from absl.flags import FLAGS
from absl import app, flags
import time
from datetime import datetime
import json
import zmq
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.set_visible_devices(physical_devices[0:1], 'GPU')

# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# deep sort imports
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-tiny-416/',
                    'path to weights file')
flags.DEFINE_string('helmet_weights', './checkpoints/super7-yolov4-tiny-416',
                    'path to weights file')
flags.DEFINE_string('clothes_weights', './checkpoints/clothes-yolov4-tiny-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4',
                    'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'directory to output video')
flags.DEFINE_string('output_format', 'XVID',
                    'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.50, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', True, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

resize_frame_width = int(os.getenv('RESIZE_FRAME_WIDTH'))
resize_frame_height = int(os.getenv('RESIZE_FRAME_HEIGHT'))

print(resize_frame_width, resize_frame_height)

context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.bind("tcp://0.0.0.0:5555")

FirebaseDB = Firebase('./FirestoreCert.json')

r = redis.Redis(host='redis', port=6379, decode_responses=True)
r.set('dangerousArea', FirebaseDB.getArea())


def getDangerousArea():
    r.set('dangerousArea', FirebaseDB.getArea())
    print("Get dangerous area: ", r.get('dangerousArea'))


def main(_argv):
    schedule.every(5).seconds.do(getDangerousArea)
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1
    peopleOut = 0
    peopleIn = 0

    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(
            FLAGS.weights, tags=[tag_constants.SERVING])
        saved_helmet_model_loaded = tf.saved_model.load(
            FLAGS.helmet_weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']
        # helmet_infer = saved_helmet_model_loaded.signatures['serving_default']
        helmetDetector = HelmetModelService(
            FLAGS.size, FLAGS.helmet_weights, FLAGS.iou, FLAGS.score)
        clothesDetector = ClothesModelService(
            FLAGS.size, FLAGS.clothes_weights, FLAGS.iou, FLAGS.score)

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None
    vid.set(cv2.CAP_PROP_FPS, 30)

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output + 'streaming_' + str(datetime.now()) +
                              '.mp4', codec, fps, (resize_frame_width, resize_frame_height))

    frame_num = 0
    # while video is running
    while True:
        schedule.run_pending()
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(
                frame, (resize_frame_width, resize_frame_height))
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num += 1
        print('Frame #: ', frame_num)

        # if (frame_num % 3 != 0):
        #     continue
        if (frame_num % 10000 == 0):
            out = cv2.VideoWriter(FLAGS.output + 'streaming_' + str(datetime.now()) +
                                  '.mp4', codec, fps, (resize_frame_width, resize_frame_height))

        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(
                output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to customize tracker for only people)
        # allowed_classes = ['helmet', 'head', 'person', 'reflective_clothes', 'other_clothes', 'with_mask', 'without_mask', 'mask_weared_incorrect']
        allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(
                count), (5, 35), 0, 0.5, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox,
                      score, class_name, feature in zip(bboxes, scores, names, features)]

        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(
            boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections, height)

        # frame = frame.copy()
        
        # Write the danger area on the frame
        areaPolygon = np.array(AreaUtils.getPolygonShape(
            json.loads(r.get('dangerousArea'))), np.int32)
        cv2.polylines(frame, [areaPolygon], isClosed=True, color=(
            0, 0, 255), thickness=3, lineType=cv2.LINE_AA)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()

            print("track.get_helmet_class: ", track.get_helmet_class())

            if track.get_helmet_class() == None or len(track.get_helmet_class()) == 0:
                print("Detect Cloth!")
                helmetDetectorResult = helmetDetector.detectHelmet(frame[int(bbox[1]):int(
                    bbox[3]), int(bbox[0]):int(bbox[2])])
                print("The detected clothes are: {}".format(helmetDetectorResult))
                helmetClassName = ''
                if len(helmetDetectorResult) > 0:
                    helmetClassName = utils.read_class_names(
                        cfg.YOLO.HELMET_CLASSES)
                    helmetClassName = helmetClassName[int(
                        helmetDetectorResult[0])]
                    print(helmetClassName)
                track.set_helmet_class(helmetClassName)
            else:
                helmetClassName = track.get_helmet_class()

            print("track.get_cloth_class: ", track.get_cloth_class())

            if track.get_cloth_class() == None or len(track.get_cloth_class()) == 0:
                print("Detect Cloth!")
                clothesDetectorResult = clothesDetector.detectClothes(frame[int(bbox[1]):int(
                    bbox[3]), int(bbox[0]):int(bbox[2])])
                print("The detected clothes are: {}".format(
                    clothesDetectorResult))
                clothesClassName = ''
                if len(clothesDetectorResult) > 0:
                    clothesClassName = utils.read_class_names(
                        cfg.YOLO.CLOTHES_CLASSES)
                    clothesClassName = clothesClassName[int(
                        clothesDetectorResult[0])]
                    print(clothesClassName)
                track.set_cloth_class(clothesClassName)
            else:
                clothesClassName = track.get_cloth_class()

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(bboxes, original_h, original_w)

            # store all predictions in one parameter for simplicity when calling functions
            pred_bbox = [bboxes, scores, classes, num_objects]

            # draw bbox on screen
            color = colors[int(track.color
                               ) % len(colors)]
            color = [i * 255 for i in color]
            circle = Utils.calculateBottomCenterCoordinate(
                int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))

            cv2.rectangle(frame, (int(bbox[0]), int(
                bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(
                len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + helmetClassName + "-" + clothesClassName + "-" + track.track_id,
                        (int(bbox[0]), int(bbox[1]-10)), 0, 0.5, (255, 255, 255), 2)
            ObjectUtils.writeCircleStatusOnFrame(
                frame, circle[0], circle[1], r.get('dangerousArea'))

            if track.stateInArea == 0 and (Utils.isInside(circle[0], circle[1], r.get('dangerousArea')) == True) and track.noConsider == False:
                cropped_image = frame[int(bbox[1]):int(
                    bbox[3]), int(bbox[0]):int(bbox[2])]

                person = {'id': track.track_id, 'image_base64': str(Utils.imageToBase64(
                    cropped_image)), 'people_in': peopleIn, 'people_out': peopleOut, 'created_at': str(datetime.now())}
                print(json.dumps([person], indent=4))

                peopleIn = peopleIn + 1
                track.stateInArea = 1
                track.noConsider = True

            elif track.stateInArea == 1 and (Utils.isInside(circle[0], circle[1], r.get('dangerousArea')) == False) and track.noConsider == False:
                cropped_image = frame[int(bbox[1]):int(
                    bbox[3]), int(bbox[0]):int(bbox[2])]

                person = {'id': track.track_id, 'image_base64': str(Utils.imageToBase64(
                    cropped_image)), 'people_in': peopleIn, 'people_out': peopleOut, 'created_at': str(datetime.now())}
                print(json.dumps([person], indent=4))

                peopleOut = peopleOut + 1
                track.stateInArea = 0
                track.noConsider = True

        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(
                    str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # cv2.imwrite("frame_%d.jpg" % frame_num, frame)

        frameBase64 = Utils.imageToBase64(result)

        jsonResult = json.dumps(
            {'frame': str(frameBase64), 'people_in': peopleIn, 'people_out': peopleOut})

        socket.send_string(jsonResult)

        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)

        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)

    # cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
