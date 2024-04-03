import logging
import os
import sys
import types
from os.path import exists

import click
import cv2
import av
import numpy as np
from dotenv import load_dotenv

from core.pipeline import fake_gpool, load_intrinsics, calibrate_and_validate_realtime, load_realtime_ref_data, map_pupil_data
from pye3d.detector_3d import ellipse2dict
from pye3d.geometry.primitives import Circle, Ellipse
from pye3d.geometry.projections import project_circle_into_image_plane

from alive_progress import alive_bar

import matplotlib.pyplot as plt

def pregenerate_eye_model(detector_3d, datums_2d, start_timestamp, stop_timestamp, threshold=None, aspectratio_threshold=None, freeze=True):
    import file_methods as fm
    prev_center = None

    for datum_2d in datums_2d:
        
        # ASPECT RATIO THRESHOLD
        if aspectratio_threshold is not None and datum_2d["ellipse"]["axes"][0] != 0.0:
            if datum_2d["ellipse"]["axes"][0] > datum_2d["ellipse"]["axes"][1]:
                aspect_ratio = datum_2d["ellipse"]["axes"][1] / datum_2d["ellipse"]["axes"][0]
            else:
                aspect_ratio = datum_2d["ellipse"]["axes"][0] / datum_2d["ellipse"]["axes"][1]
            if aspect_ratio > aspectratio_threshold:
                datum_2d = datum_2d._deep_copy_dict()
                datum_2d["confidence"] = 0.75
                datum_2d = fm.Serialized_Dict(python_dict=datum_2d)
        
        # VELOCITY THRESHOLD
        if threshold is not None:
            if prev_center is not None:
                delta = np.linalg.norm(np.array(datum_2d["ellipse"]["center"]) - np.array(prev_center))
            else:
                delta = -1  # Cannot assume that the first detected pupil ellipse in a series is accurate, must get at least 2
            prev_center = datum_2d["ellipse"]["center"]
            
            if delta >= 0 and delta <= threshold and datum_2d["timestamp"] >= start_timestamp and datum_2d["timestamp"] <= stop_timestamp:
                observation = detector_3d._extract_observation(datum_2d)
                detector_3d.update_models(observation)
                datum_2d = datum_2d._deep_copy_dict()
                datum_2d["confidence"] = 0.65
                datum_2d = fm.Serialized_Dict(python_dict=datum_2d)
        
        # UPDATE MODELS
        if datum_2d["timestamp"] >= start_timestamp and datum_2d["timestamp"] <= stop_timestamp:
            observation = detector_3d._extract_observation(datum_2d)
            detector_3d.update_models(observation)

    if freeze:
        setattr(detector_3d, "is_long_term_model_frozen", True)
    return freeze

from scipy.optimize import minimize

def test_calib_acc(offset, detector3d,#D3D, g_pool, datum_list, start_model_timestamp, freeze_model_timestamp, threshold, Aspect_ratio_threshold, freeze,
    old_sphere_center, recording_path, relevant_calib_frames, relevant_datum_2ds, mapping_method, 
    count_obj=None):

    from operator import itemgetter
    import file_methods as fm
    from core.pipeline import load_pupil_data

    #detector3d = D3D(g_pool)
    #pregenerate_eye_model(detector3d.pupil_detector, datum_list, start_model_timestamp, freeze_model_timestamp, threshold, aspectratio_threshold=Aspect_ratio_threshold, freeze=freeze)
    
    # MUL BY 1.1
    detector3d.pupil_detector.long_term_model.set_sphere_center(np.array(old_sphere_center) - 100 + np.array(offset))

    # STEP 1: Map gaze data for calibration points only
    rt_ref_data = load_realtime_ref_data(recording_path[:-9]+'/realtime_calib_points.msgpack')
    rt_datums = []
    container = av.open(recording_path)
    stream = container.streams.video[0]

    for _, (pupil_frame, datum_2d) in enumerate(zip(relevant_calib_frames, relevant_datum_2ds)):
        pupil3d_datum = detector3d.detect(
            pupil_frame, **{"previous_detection_results": [datum_2d]}
        )
        pupil3d_datum['topic'] = 'pupil.0.3d'
        pupil3d_datum['id'] = 0
        rt_datums.append(fm.Serialized_Dict(python_dict=pupil3d_datum))
        pupil3d_datum['topic'] = 'pupil.1.3d'
        pupil3d_datum['id'] = 1
        rt_datums.append(fm.Serialized_Dict(python_dict=pupil3d_datum))
    intrinsics_loc = recording_path[:-9] + "/world.intrinsics"
    rt_data = types.SimpleNamespace()
    rt_data.data = rt_datums
    rt_data_real = load_pupil_data(recording_path[:-9] + f"/offline_data/vanilla/offline_pupil.pldata")
    calibrated_gazer, pupil_data = calibrate_and_validate_realtime(rt_ref_data, rt_data, intrinsics_loc, mapping_method, min_calibration_confidence=0.0)
    gaze, gaze_ts = map_pupil_data(calibrated_gazer, pupil_data, bar_enabled=False)
    
    # STEP 2: Calculate calibration point accuracy
    acc_errors = []
    curr_gaze_idx = 1
    for curr_rt_data in rt_ref_data:
        timestamp = curr_rt_data['timestamp']
        satisfied = False
        while not satisfied:
            if gaze[curr_gaze_idx-1]['timestamp'] < timestamp and gaze[curr_gaze_idx]['timestamp'] >= timestamp:
                rt_ref_data_az = np.rad2deg(np.arctan2(curr_rt_data['screen_pos'][0],curr_rt_data['screen_pos'][2]))
                rt_ref_data_el = np.rad2deg(np.arctan2(curr_rt_data['screen_pos'][1],curr_rt_data['screen_pos'][2]))
                try:
                    gaze_normal_2 = (gaze[curr_gaze_idx]['gaze_normals_3d']['0'][0] + gaze[curr_gaze_idx]['gaze_normals_3d']['1'][0],
                        gaze[curr_gaze_idx]['gaze_normals_3d']['0'][1] + gaze[curr_gaze_idx]['gaze_normals_3d']['1'][1],
                        gaze[curr_gaze_idx]['gaze_normals_3d']['0'][2] + gaze[curr_gaze_idx]['gaze_normals_3d']['1'][2])
                except KeyError:
                    #print("NO VALID GAZE POINT")
                    satisfied = True
                    continue
                gaze_normal_2 = [f/2.0 for f in gaze_normal_2]
                gaze_normal_az = np.rad2deg(np.arctan2(gaze_normal_2[0],gaze_normal_2[2]))
                gaze_normal_el = np.rad2deg(np.arctan2(gaze_normal_2[1],gaze_normal_2[2]))
                gaze_normal_el = -gaze_normal_el
                acc_error = np.linalg.norm(np.array([rt_ref_data_az, rt_ref_data_el]) - np.array([gaze_normal_az, gaze_normal_el]))
                acc_errors.append(acc_error)
                satisfied = True
            else:
                curr_gaze_idx += 1
    result = np.mean(acc_errors)
    #if count_obj: 
    #    logging.info("[{}]{} offset: {} acc error".format(count_obj.get(), offset, result))
    #else:
    #    logging.info("{} offset: {} acc error".format(offset, result))
    return result


def test_calib_iou(offset, detector3d,#D3D, g_pool, datum_list, start_model_timestamp, freeze_model_timestamp, threshold, Aspect_ratio_threshold, freeze,
    old_sphere_center, recording_path, relevant_calib_frames, relevant_datum_2ds, mapping_method, 
    count_obj=None):
    
    # MUL BY 1.1
    detector3d.pupil_detector.long_term_model.set_sphere_center(np.array(old_sphere_center) - 100 + np.array(offset))

    reproj_errors_centroid = []
    reproj_errors_IoU = []
    reproj_errors_angle = []
    reproj_errors_axes = []
    reproj_errors_all_props = []

    for _, (pupil_frame, datum_2d) in enumerate(zip(relevant_calib_frames, relevant_datum_2ds)):
        # Maybe skip every 2 or 3 or 4 or (etc) frames to cut down on processing time.
        pupil3d_datum = detector3d.detect(
            pupil_frame, **{"previous_detection_results": [datum_2d]}
        )
        
        # Centroid Error
        ellipse_2d = datum_2d["ellipse"]
        proj_ellipse = pupil3d_datum["circle_3d"]
        proj_circle = Circle(proj_ellipse['center'], proj_ellipse['normal'], proj_ellipse['radius'])
        camera_ = detector3d.detector.camera
        proj_ellipse = project_circle_into_image_plane(
            proj_circle,
            focal_length=camera_.focal_length,
            transform=True,
            width=camera_.resolution[0],
            height=camera_.resolution[1],
        )
        if not proj_ellipse:
            proj_ellipse = Ellipse(np.asarray([0.0, 0.0]), 0.0, 0.0, 0.0)
        proj_ellipse = ellipse2dict(proj_ellipse)
        #reproj_error = np.linalg.norm(np.array(proj_ellipse["center"]) - np.array(datum_2d["ellipse"]["center"]))
        reproj_error = np.linalg.norm(np.array(proj_ellipse["center"]) - np.array(ellipse_2d["center"]))
        reproj_errors_centroid.append(reproj_error)

        # IoU
        height = camera_.resolution[1]
        width = camera_.resolution[0]
        detected_ell_im = cv2.ellipse(np.zeros((height, width)).astype(np.uint8), (int(ellipse_2d["center"][0]), int(ellipse_2d["center"][1])),
            (int(ellipse_2d["axes"][0] / 2), int(ellipse_2d["axes"][1] / 2)),
            ellipse_2d["angle"], 0., 360., 255, -1)
        reprojected_ell_im = cv2.ellipse(np.zeros((height, width)).astype(np.uint8), (int(proj_ellipse["center"][0]), int(proj_ellipse["center"][1])),
            (int(proj_ellipse["axes"][0] / 2), int(proj_ellipse["axes"][1] / 2)),
            proj_ellipse["angle"], 0., 360., 255, -1)
        intersection_mask = cv2.bitwise_and(detected_ell_im, reprojected_ell_im)
        union_mask = cv2.bitwise_or(detected_ell_im, reprojected_ell_im)
        intersection = cv2.countNonZero(intersection_mask)
        union = cv2.countNonZero(union_mask)
        if union > 0:
            IoU = intersection / union
        else:
            IoU = 0.0
        reproj_errors_IoU.append(1.0 - IoU)
        
        # Angle Error
        angle1 = ellipse_2d["angle"]
        angle2 = proj_ellipse["angle"]
        difang = angle1 - angle2
        if difang > 180.:
            difang = difang - 360.
        elif difang < -180.:
            difang = difang + 360.
        reproj_errors_angle.append(np.abs(difang))

        # Axes error
        if ellipse_2d["axes"][0] > proj_ellipse["axes"][0]:
            axes1_diff = proj_ellipse["axes"][0] / ellipse_2d["axes"][0]
        else:
            axes1_diff = ellipse_2d["axes"][0] / proj_ellipse["axes"][0]

        if ellipse_2d["axes"][1] > proj_ellipse["axes"][1]:
            axes2_diff = proj_ellipse["axes"][1] / ellipse_2d["axes"][1]
        else:
            axes2_diff = ellipse_2d["axes"][1] / proj_ellipse["axes"][1]
        reproj_errors_axes.append(np.mean([1.0 - axes1_diff, 1.0 - axes2_diff]))
        
        BOUNDED_SCALAR = 10.0
        reproj_errors_all_props.append( (np.mean([1.0 - axes1_diff, 1.0 - axes2_diff]))*BOUNDED_SCALAR + np.abs(difang) + reproj_error)

    #result = np.mean(reproj_errors_centroid)
    result = np.mean(reproj_errors_IoU)
    #result = np.mean(reproj_errors_angle)
    #result = np.mean(reproj_errors_axes)
    #result = np.mean(reproj_errors_all_props)

    with open("minimize_results.csv", "a") as outfile:
        if count_obj:
            inc = count_obj.get()
        else:
            inc = ""
        outfile.write(str(inc)+","+str(result)+","+str(offset)+"\n")
        print("["+str(inc)+"]: "+str(result)+",    "+str(offset))

    #if count_obj: 
    #    logging.info("[{}]{} offset: {} acc error".format(count_obj.get(), offset, result))
    #else:
    #    logging.info("{} offset: {} acc error".format(offset, result))
    return result


def minimize_myfunc(args):
    MAXITER = 5000  # 10
    fun, inp, a = args
    D3D, g_pool, datum_list, start_model_timestamp, freeze_model_timestamp, threshold, Aspect_ratio_threshold, freeze = a[:8]

    #sys.stdout = open(os.devnull, 'w')
    #logging.disable(logging.CRITICAL)

    detector3d = D3D(g_pool)
    pregenerate_eye_model(detector3d.pupil_detector, datum_list, start_model_timestamp, freeze_model_timestamp, threshold, aspectratio_threshold=Aspect_ratio_threshold, freeze=freeze)
    b = (detector3d, *(a[8:]))
    #a[7] = detector3d
    res = minimize(fun, np.array([100.0, 100.0, 100.0])+inp, args=b, method='BFGS',#method='nelder-mead',#method='BFGS',#method='nelder-mead',
        options={'maxiter': MAXITER, 'eps': 0.01})#'xatol': 0.001})

    #sys.stdout = sys.__stdout__
    #logging.disable(logging.NOTSET)
    return (res.x, fun(res.x, *b))


def pl_detection_on_video(recording_path, g_pool, pupil_params, detector_plugin=None, debug_window=False, delta_fig_path=None, load_2d_pupils=False, skip_3d_detection=False, start_model_timestamp=None, freeze_model_timestamp=None, mapping_method=None):
    IOU_threshold = 0.98
    Delta_threshold = 85.0
    Aspect_ratio_threshold = 0.8
    freeze_model = True

    flip = recording_path[-5] == "1"  # eye1
    roi = None
    detector2d = None
    detector3d = None
    timestamps_path = recording_path[0 : recording_path.rindex(".")] + "_timestamps.npy"
    timestamps = np.load(timestamps_path)
    id = int(recording_path[0 : recording_path.rindex(".")][-1])
    topic = str(id)

    import file_methods as fm
    if detector_plugin is None:
        from pupil_detector_plugins.detector_2d_plugin import Detector2DPlugin
        VanillaDetector2DPlugin = Detector2DPlugin
    else:
        Detector2DPlugin = detector_plugin
        from pupil_detector_plugins.detector_2d_plugin import Detector2DPlugin as VanillaDetector2DPlugin
    from pupil_detector_plugins.pye3d_plugin import Pye3DPlugin
    from roi import Roi

    #vidcap = cv2.VideoCapture(recording_path)
    container = av.open(recording_path)
    stream = container.streams.video[0]
    #total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_frames = container.streams.video[0].frames

    datum_list = {"2d": [], "3d": [], "debug_imgs":[]}
    #success, frame = vidcap.read()

    #height, width = frame[:, :, 0].shape
    for idx, frame in enumerate(container.decode(stream)):
        width = frame.width
        height = frame.height
        break

    from plugin import Plugin_List
    #g_pool.app = "capture"  # SETS PY3D DETECTOR TO REALTIME ("asynchronous") MODE
    g_pool.display_mode = "n/a"
    g_pool.eye_id = id
    g_pool.plugin_by_name = []
    g_pool.plugins = Plugin_List(
        g_pool, []
    )
    roi = Roi(
        g_pool=g_pool, frame_size=(width, height), bounds=(0, 0, width, height)
    )

    detector2d = Detector2DPlugin(g_pool=g_pool, properties=pupil_params[id])
    if not skip_3d_detection:
        detector3d = Pye3DPlugin(g_pool=g_pool)
    
    count = 0
    plugin_name = "vanilla"
    plconfs = []
    centerdelta = []
    thetadelta = []
    axesdelta = []
    axesratio = []
    ious = []
    reproj_errors_centroid = []
    reproj_errors_axes = []
    prev_center = None
    prev_theta = None
    prev_axes = None
    im_prev = None
    if detector_plugin is not None:
        plugin_name = detector_plugin.__name__
    recording_loc = recording_path[:-9]
    file_path = recording_loc + "/offline_data/"+plugin_name
    if load_2d_pupils and exists(file_path+"/offline_pupil.pldata"):
        logging.info(f"Loading previously detected pupil datums of {recording_path} ({plugin_name})")
        with alive_bar(int(total_frames), bar = "filling") as bar:
            import player_methods as pm
            pupil_datums_bisector = pm.PupilDataBisector.load_from_file(file_path, "offline_pupil")
            for loaded_datum in pupil_datums_bisector[recording_path[-5],"2d"]:
                #if not success:
                #    break
                # PL confidence
                plconfs.append(loaded_datum["confidence"])
                if prev_center is not None:
                    # center delta
                    center_delta = np.linalg.norm(np.array(loaded_datum["ellipse"]["center"]) - np.array(prev_center))
                    if center_delta <= np.linalg.norm(np.array((height,width))):
                        centerdelta.append(center_delta)
                    # theta delta
                    theta_delta = loaded_datum["ellipse"]["angle"] - prev_theta
                    theta_delta = (theta_delta + 180) % 360 - 180
                    thetadelta.append(theta_delta)
                    # axes delta
                    axes_delta = np.linalg.norm(np.array(loaded_datum["ellipse"]["axes"]) - np.array(prev_axes))
                    axesdelta.append(axes_delta)
                    # axes ratio
                    if loaded_datum["ellipse"]["axes"][1] != 0.0:
                        axesratio.append(loaded_datum["ellipse"]["axes"][1] / loaded_datum["ellipse"]["axes"][0])
                    # IoU
                    im_curr = cv2.ellipse(np.zeros((height, width)).astype(np.uint8), (int(loaded_datum["ellipse"]["center"][0]), int(loaded_datum["ellipse"]["center"][1])),
                        (int(loaded_datum["ellipse"]["axes"][0] / 2), int(loaded_datum["ellipse"]["axes"][1] / 2)),
                        loaded_datum["ellipse"]["angle"], 0., 360., 255, -1)
                    if im_prev is None:
                        im_prev = cv2.ellipse(np.zeros((height, width)).astype(np.uint8), (int(prev_center[0]), int(prev_center[1])),
                            (int(prev_axes[0] / 2), int(prev_axes[1] / 2)),
                            prev_theta, 0., 360., 255, -1)
                    intersection_mask = cv2.bitwise_and(im_curr, im_prev)
                    union_mask = cv2.bitwise_or(im_curr, im_prev)
                    IoU = cv2.countNonZero(intersection_mask) / cv2.countNonZero(union_mask)
                    im_prev = im_curr
                    ious.append(IoU)
                    # OVERRIDE CONFIDENCE
                    if IOU_threshold is not None and IoU <= IOU_threshold:
                        loaded_datum = loaded_datum._deep_copy_dict()
                        loaded_datum["confidence"] = 0.75#IoU
                        loaded_datum = fm.Serialized_Dict(python_dict=loaded_datum)
                
                prev_center = loaded_datum["ellipse"]["center"]
                prev_theta = loaded_datum["ellipse"]["angle"]
                prev_axes = loaded_datum["ellipse"]["axes"]
                datum_list["2d"].append(loaded_datum)
                #success, frame = vidcap.read()
                bar()
    else:
        logging.info(f"{recording_path} ({plugin_name}) 2D:")
        with alive_bar(int(total_frames), bar = "filling") as bar:
            # Pass 1 - 2D detector
            #while success:
            for idx, avframe in enumerate(container.decode(stream)):
                frame = np.array(avframe.to_image())
                timestamp = timestamps[count]
                bgr = frame#[:, :]#, 0]
                bgr = bgr.copy(order="C")
                gray = frame[:, :, 0]
                gray = gray.copy(order="C")
                height, width = gray.shape

                pupil_frame = lambda: None
                setattr(pupil_frame, "gray", gray)
                setattr(pupil_frame, "bgr", bgr)
                setattr(pupil_frame, "width", width)
                setattr(pupil_frame, "height", height)
                setattr(pupil_frame, "timestamp", timestamp)
                pupil_datum = detector2d.detect(pupil_frame)
                pupil_datum = fm.Serialized_Dict(python_dict=pupil_datum)
                # PL confidence
                plconfs.append(pupil_datum["confidence"])
                
                if pupil_datum["confidence"] > 0.98:
                    cv2.imwrite(f"{recording_loc}/above_0.98/{id}_{idx}.png", bgr)
                
                if prev_center is not None:
                    # center delta
                    center_delta = np.linalg.norm(np.array(pupil_datum["ellipse"]["center"]) - np.array(prev_center))
                    if center_delta <= np.linalg.norm(np.array((height,width))):
                        centerdelta.append(center_delta)
                    # theta delta
                    theta_delta = pupil_datum["ellipse"]["angle"] - prev_theta
                    theta_delta = (theta_delta + 180) % 360 - 180
                    thetadelta.append(theta_delta)
                    # axes delta
                    axes_delta = np.linalg.norm(np.array(pupil_datum["ellipse"]["axes"]) - np.array(prev_axes))
                    axesdelta.append(axes_delta)
                    # axes ratio
                    if pupil_datum["ellipse"]["axes"][1] != 0.0:
                        axesratio.append(pupil_datum["ellipse"]["axes"][1] / pupil_datum["ellipse"]["axes"][0])
                    # IoU
                    try:
                        im_curr = cv2.ellipse(np.zeros((height, width)).astype(np.uint8), (int(pupil_datum["ellipse"]["center"][0]), int(pupil_datum["ellipse"]["center"][1])),
                            (int(pupil_datum["ellipse"]["axes"][0] / 2), int(pupil_datum["ellipse"]["axes"][1] / 2)),
                            pupil_datum["ellipse"]["angle"], 0., 360., 255, -1)
                        if im_prev is None:
                            im_prev = cv2.ellipse(np.zeros((height, width)).astype(np.uint8), (int(prev_center[0]), int(prev_center[1])),
                                (int(prev_axes[0] / 2), int(prev_axes[1] / 2)),
                                prev_theta, 0., 360., 255, -1)
                        intersection_mask = cv2.bitwise_and(im_curr, im_prev)
                        union_mask = cv2.bitwise_or(im_curr, im_prev)
                        IoU = cv2.countNonZero(intersection_mask) / cv2.countNonZero(union_mask)
                        im_prev = im_curr
                    except:     
                        IoU = 0.0
                    ious.append(IoU)
                    # OVERRIDE CONFIDENCE
                    if IOU_threshold is not None and IoU <= IOU_threshold:
                        pupil_datum = pupil_datum._deep_copy_dict()
                        pupil_datum["confidence"] = 0.75#IoU
                        pupil_datum = fm.Serialized_Dict(python_dict=pupil_datum)
                
                prev_center = pupil_datum["ellipse"]["center"]
                prev_theta = pupil_datum["ellipse"]["angle"]
                prev_axes = pupil_datum["ellipse"]["axes"]
                #pupil3d_datum = detector3d.detect(
                #    pupil_frame, **{"previous_detection_results": [pupil_datum]}
                #)
                datum_list["2d"].append(pupil_datum)
                #datum_list["3d"].append(fm.Serialized_Dict(python_dict=pupil3d_datum))
                
                #datum_list["debug_imgs"].append(pupil_frame.bgr)#(frame)
                #if debug_window:
                #    cv2.imshow("debug", pupil_frame.bgr)
                #    cv2.waitKey(1)
                count += 1
                #if count % 1000 == 0:
                #    logging.info(f"{count}/{int(total_frames)}")
                bar()
                #success, frame = vidcap.read()
    
    #np.histogram(centerdelta, bins=np.ceil(np.sqrt(height**2 + width**2)) * 2)
    plt.hist(centerdelta, bins=int(np.ceil(np.sqrt(height**2 + width**2)) * 2), histtype='step',edgecolor='r',linewidth=3)
    H, bins = np.histogram(centerdelta, bins=int(np.ceil(np.sqrt(height**2 + width**2)) * 2))
    H+=np.histogram(centerdelta, bins=int(np.ceil(np.sqrt(height**2 + width**2)) * 2))[0]

    plt.bar(bins[:-1],H,width=1)

    plt.title("Pupil Centroid Celta")
    #plt.show()
    plt.clf()
    
    plt.hist(thetadelta, bins=int(360 * 8), histtype='step',edgecolor='r',linewidth=3)
    H, bins = np.histogram(thetadelta, bins=int(360 * 8))
    H+=np.histogram(thetadelta, bins=int(360 * 8))[0]
    plt.bar(bins[:-1],H,width=1)
    plt.title("Pupil Theta Delta")
    #plt.show()
    plt.clf()
    
    plt.hist(axesdelta, bins=int(np.max(axesdelta))*4, histtype='step',edgecolor='r',linewidth=3)
    H, bins = np.histogram(axesdelta, bins=int(np.max(axesdelta))*4)
    H+=np.histogram(axesdelta, bins=int(np.max(axesdelta))*4)[0]
    plt.bar(bins[:-1],H,width=1)
    plt.title("Pupil Axes Delta")
    #plt.show()
    plt.clf()
    
    plt.hist(axesratio, bins=1000, histtype='step',edgecolor='r',linewidth=3)
    #H, bins = np.histogram(axesratio, bins=1000)
    #H+=np.histogram(axesratio, bins=1000)[0]
    #plt.bar(bins[:-1],H,width=1)
    plt.title("Pupil Axes Ratio")
    #plt.show()
    plt.clf()
    #exit()
    
    if True or Delta_threshold is None:
        threshold = None
    else:
        threshold = np.percentile(centerdelta, Delta_threshold)  # x% least moving pupil centroids

    if not skip_3d_detection:
        # Generate model based on entire video
        if freeze_model:
            from operator import itemgetter
            start_model_timestamp = min(datum_list["2d"], key=itemgetter("timestamp"))["timestamp"]#0.0
            freeze_model_timestamp = max(datum_list["2d"], key=itemgetter("timestamp"))["timestamp"]
        else:
            start_model_timestamp = None
            freeze_model_timestamp = None
        
        has_frozen = False
        #import copy
        #sys.setrecursionlimit(10000)
        #temp_list = []
        #for datum in datum_list["2d"]:
        #    temp_list.append(datum["confidence"])

        sync_updates_PL = False
        
        if not sync_updates_PL and (start_model_timestamp is not None and freeze_model_timestamp is not None):
            logging.info(f"Pregenerating 3D eye model...")
            has_frozen = pregenerate_eye_model(detector3d.pupil_detector, datum_list["2d"], start_model_timestamp, freeze_model_timestamp, threshold, aspectratio_threshold=Aspect_ratio_threshold, freeze=True)
            
            ML_OPTIMIZE_SPHERE_POS = False
            
            if ML_OPTIMIZE_SPHERE_POS:
                old_sphere_center = detector3d.pupil_detector.long_term_model.sphere_center

                relevant_calib_frames = []
                relevant_datum_2ds = []
                logging.info("Pre-fetching calibration frames from eye video...")
                rt_ref_data = load_realtime_ref_data(recording_path[:-9]+'/realtime_calib_points.msgpack')
                min_calib_timestamp = min(rt_ref_data, key=itemgetter("timestamp"))["timestamp"] - 1  # 1-second buffer
                max_calib_timestamp = max(rt_ref_data, key=itemgetter("timestamp"))["timestamp"] + 1  # 1-second buffer
                with alive_bar(int(total_frames), bar = "filling") as bar:
                    for idx, (datum_2d, avframe) in enumerate(zip(datum_list["2d"], container.decode(stream))):
                        if datum_2d["timestamp"] >= min_calib_timestamp and datum_2d["timestamp"] <= max_calib_timestamp:
                            frame = np.array(avframe.to_image())
                            frame = frame.copy(order="C")#datum_list["debug_imgs"][idx]
                            timestamp = timestamps[idx]
                            bgr = frame[:, :, 0]
                            bgr = bgr.copy(order="C")
                            gray = frame[:, :, 0]
                            gray = gray.copy(order="C")
                            height, width = gray.shape
                            pupil_frame = types.SimpleNamespace()#lambda: None
                            setattr(pupil_frame, "gray", gray)
                            setattr(pupil_frame, "bgr", bgr)
                            setattr(pupil_frame, "width", width)
                            setattr(pupil_frame, "height", height)
                            setattr(pupil_frame, "timestamp", timestamp)
                            relevant_calib_frames.append(pupil_frame)
                            relevant_datum_2ds.append(datum_2d)
                        bar()
                        if datum_2d["timestamp"] > max_calib_timestamp:
                            break

                class Count_Obj:
                    def __init__(self):
                        self.count = 0
                    def get(self):
                        self.count += 1
                        return self.count

                count_obj = Count_Obj()

                # nelder-mead best at 50: 3.6883709901275483
                # bfgs best at 10: 3.7477321875731393

                # nelder-mead at 500 with 100 offset: [101.02694368 102.3209565  103.57146832] (1.7892887046276487)
                # nelder-mead at 50 with 100 offset: [101.02687722 102.32098364 103.57511868] (1.7873042425587338)
                # nelder-mead at 25 with 100 offset: [101.04564053, 102.39952554, 104.87536951] (1.8232083813850164)
                # nelder-mead at 13*13*13 with 100 offset: WINNER:  1.7911638474522695 [101.075      102.36666667 104.19131944]
                # nelder-mead at 7*7*7 with 100 offset: WINNER:  1.810530914234763 [100.83868313 102.32537723 104.33429355]
                # nelder-mead at 5*5*5 with 100 offset: WINNER:  1.807191536087017 [101.09375    102.42578125 103.83463542]
                #ARGS = list(np.array([[g//(13*13), (g%(13*13))//13, g%13] for g in range(13*13*13)]) - 6)

                from multiprocessing import Pool, cpu_count
                CORES = int(cpu_count() * 0.50)
                logging.info(f"Optimizing 3D eye model positions on {CORES} cores. This may take a while. {recording_path} ({plugin_name})")
                ARGS = list(np.array([[(g//(5*5))*2.5, ((g%(5*5))//5)*2.5, (g%5)*2.5] for g in range(5*5*5)]) - 5)
                g_pool.pupil_detector = None
                g_pool.roi = None
                
                #ARGS = [(test_calib_acc, g, (Pye3DPlugin, g_pool, datum_list["2d"], start_model_timestamp, freeze_model_timestamp, threshold, Aspect_ratio_threshold, True,
                #    old_sphere_center, recording_path, relevant_calib_frames, relevant_datum_2ds, mapping_method,
                #    None)) for g in ARGS]
                
                ARGS = [(test_calib_iou, g, (Pye3DPlugin, g_pool, datum_list["2d"], start_model_timestamp, freeze_model_timestamp, threshold, Aspect_ratio_threshold, True,
                    old_sphere_center, recording_path, relevant_calib_frames, relevant_datum_2ds, mapping_method,
                    None)) for g in ARGS]
                #test_calib_acc(offset, detector3d, old_sphere_center, recording_path, relevant_calib_frames, relevant_datum_2ds, mapping_method, 
                #    count_obj=None)

                """
                # ML adjustment
                def mute():
                    sys.stdout = open(os.devnull, 'w')
                    logging.disable(logging.CRITICAL)
                p = Pool(CORES, maxtasksperchild=1, initializer=mute)
                results = p.map(minimize_myfunc, ARGS)
                p.close()
                curr_min = None
                for result in results:
                    if curr_min is None or result[1] < curr_min[1]:
                        curr_min = result
                print("WINNER: ", curr_min[1], curr_min[0])
                detector3d.pupil_detector.long_term_model.set_sphere_center(np.array(old_sphere_center) - 100 + np.array(curr_min[0]))
                """
                
                """
                z_z_z = test_calib_acc([0.0, 0.0, 0.0])
                o_o_o = test_calib_acc([1.0, 1.0, 1.0])
                t_t_t = test_calib_acc([2.0, 2.0, 2.0])
                tr_tr_tr = test_calib_acc([3.0, 3.0, 3.0])
                print("[0.0, 0.0, 0.0] offset mean accuracy error:")
                print(z_z_z, "degrees")
                print()
                print("[1.0, 1.0, 1.0] offset mean accuracy error:")
                print(o_o_o, "degrees")
                print()
                print("[2.0, 2.0, 2.0] offset mean accuracy error:")
                print(t_t_t, "degrees")
                print()
                print("[3.0, 3.0, 3.0] offset mean accuracy error:")
                print(tr_tr_tr, "degrees")
                exit()
                """
                
                start_point = -11.0
                result = ([0., 0., 0.], 1.0)
                while result[1] == 1.0:
                    start_point = start_point + 1.0
                    result = minimize_myfunc((test_calib_iou, [start_point,start_point,start_point], (Pye3DPlugin, g_pool, datum_list["2d"], start_model_timestamp, freeze_model_timestamp, threshold, Aspect_ratio_threshold, True,
                        old_sphere_center, recording_path, relevant_calib_frames, relevant_datum_2ds, mapping_method,
                        Count_Obj())))

                print("WINNER: ", result[1], result[0])
                detector3d.pupil_detector.long_term_model.set_sphere_center(np.array(old_sphere_center) - 100 + np.array(result[0]))

        #for i in range(len(datum_list["2d"])):
        #    datum_list["2d"][i]["confidence"] = temp_list[i]
        #datum_list["2d"] = temp_list

        #vidcap = cv2.VideoCapture(recording_path)
        #success, frame = vidcap.read()
        container = av.open(recording_path)
        stream = container.streams.video[0]
        logging.info(f"{recording_path} ({plugin_name}) 3D:")
        prev_center = None

        if sync_updates_PL:
            new_detector2d = VanillaDetector2DPlugin(g_pool=g_pool, properties=pupil_params[id])

        with alive_bar(int(total_frames), bar = "filling") as bar:
            # Pass 2 - 3D detector
            for idx, (datum_2d, avframe) in enumerate(zip(datum_list["2d"], container.decode(stream))):
                frame = np.array(avframe.to_image())
                frame = frame.copy(order="C")#datum_list["debug_imgs"][idx]
                timestamp = timestamps[idx]
                bgr = frame[:, :, 0]
                bgr = bgr.copy(order="C")
                gray = frame[:, :, 0]
                gray = gray.copy(order="C")
                height, width = gray.shape
                
                if prev_center is not None:
                    delta = np.linalg.norm(np.array(datum_2d["ellipse"]["center"]) - np.array(prev_center))
                else:
                    delta = -1  # Cannot assume that the first detected pupil ellipse in a series is accurate, must get at least 2
                prev_center = datum_2d["ellipse"]["center"]
                
                # ASPECT RATIO THRESHOLD
                if Aspect_ratio_threshold is not None and datum_2d["ellipse"]["axes"][0] != 0.0:
                    if datum_2d["ellipse"]["axes"][0] > datum_2d["ellipse"]["axes"][1]:
                        aspect_ratio = datum_2d["ellipse"]["axes"][1] / datum_2d["ellipse"]["axes"][0]
                    else:
                        aspect_ratio = datum_2d["ellipse"]["axes"][0] / datum_2d["ellipse"]["axes"][1]
                    if aspect_ratio > Aspect_ratio_threshold:
                        datum_2d = datum_2d._deep_copy_dict()
                        datum_2d["confidence"] = 0.75
                        datum_2d = fm.Serialized_Dict(python_dict=datum_2d)
                
                # VELOCITY THRESHOLD
                if threshold is not None and delta > threshold:
                    datum_2d = datum_2d._deep_copy_dict()
                    datum_2d["confidence"] = 0.65
                    datum_2d = fm.Serialized_Dict(python_dict=datum_2d)
                
                # FREEZE MODEL AT PROVIDED TIMESTAMP
                if freeze_model_timestamp is not None and has_frozen is False and timestamp >= freeze_model_timestamp:
                    #detector3d.is_long_term_model_frozen = True
                    setattr(detector3d.pupil_detector, "is_long_term_model_frozen", True)
                    #detector3d.pupil_detector._long_term_schedule.pause()
                    #detector3d.pupil_detector._ult_long_term_schedule.pause()
                    has_frozen = True
                    logging.info("Freezing eye model at timestamp " + str(timestamp))
                
                """
                # PUT MASKS INTO 3D DETECTOR INSTEAD OF THE INPUT IMAGE
                if g_pool.eye_id==1:
                    res = detector2d_.detect(np.flip(gray, axis=0))
                else:
                    res = detector2d_.detect(gray)
                if res:
                    seg_map = res[0]
                    if g_pool.eye_id==1:
                        seg_map = np.flip(seg_map, axis=0)
                    seg_map[np.where(seg_map == 0)] = 255
                    seg_map[np.where(seg_map == 1)] = 128
                    seg_map[np.where(seg_map == 2)] = 0
                    res = np.array(seg_map, dtype=np.uint8)
                else:
                    res = gray
                """
                
                if sync_updates_PL:
                    new_2d_pupil_frame = lambda: None
                    setattr(new_2d_pupil_frame, "gray", gray)
                    setattr(new_2d_pupil_frame, "bgr", bgr)
                    setattr(new_2d_pupil_frame, "width", width)
                    setattr(new_2d_pupil_frame, "height", height)
                    setattr(new_2d_pupil_frame, "timestamp", timestamp)
                    new_datum_2d = new_detector2d.detect(new_2d_pupil_frame)
                    datum_2d = datum_2d._deep_copy_dict()
                    datum_2d["confidence"] = new_datum_2d["confidence"]
                    datum_2d = fm.Serialized_Dict(python_dict=datum_2d)
                
                pupil_frame = lambda: None
                setattr(pupil_frame, "gray", gray)
                setattr(pupil_frame, "bgr", bgr)
                setattr(pupil_frame, "width", width)
                setattr(pupil_frame, "height", height)
                setattr(pupil_frame, "timestamp", timestamp)

                # MUL BY 1.1
                #detector3d.pupil_detector.long_term_model.set_sphere_center(old_sphere_center * 1.1)
                pupil3d_datum = detector3d.detect(
                    pupil_frame, **{"previous_detection_results": [datum_2d]}
                )
                datum_list["3d"].append(fm.Serialized_Dict(python_dict=pupil3d_datum))
                
                # Calculate reprojection error
                proj_ellipse = pupil3d_datum["circle_3d"]#["ellipse"]
                proj_circle = Circle(proj_ellipse['center'], proj_ellipse['normal'], proj_ellipse['radius'])
                
                camera_ = detector3d.detector.camera
                proj_ellipse = project_circle_into_image_plane(
                    proj_circle,
                    focal_length=camera_.focal_length,
                    transform=True,
                    width=camera_.resolution[0],
                    height=camera_.resolution[1],
                )
                if not proj_ellipse:
                    proj_ellipse = Ellipse(np.asarray([0.0, 0.0]), 0.0, 0.0, 0.0)
                proj_ellipse = ellipse2dict(proj_ellipse)
                #reproj_error = np.linalg.norm(np.array(proj_ellipse["center"]) - np.array(datum_2d["ellipse"]["center"]))
                
                reproj_error = np.linalg.norm(np.array(proj_ellipse["center"]) - np.array(datum_2d["ellipse"]["center"]))
                reproj_errors_centroid.append(reproj_error)
                
                try:
                    if proj_ellipse["axes"][0] > datum_2d["ellipse"]["axes"][0]:
                        first = datum_2d["ellipse"]["axes"][0] / proj_ellipse["axes"][0]
                    else:
                        first = proj_ellipse["axes"][0] / datum_2d["ellipse"]["axes"][0]
                        
                    if proj_ellipse["axes"][1] > datum_2d["ellipse"]["axes"][1]:
                        second = datum_2d["ellipse"]["axes"][1] / proj_ellipse["axes"][1]
                    else:
                        second = proj_ellipse["axes"][1] / datum_2d["ellipse"]["axes"][1]
                    reproj_error = 0.5*first + 0.5*second
                except ZeroDivisionError as e:
                    reproj_error = 0.0
                reproj_errors_axes.append(1 - reproj_error)

                if debug_window:
                    #proj_pupilpts = cv2.ellipse2Poly(
                    #            center=(int(proj_ellipse["ellipse"]["center"][0]), height - int(proj_ellipse["ellipse"]["center"][1])),
                    #            axes=(int(proj_ellipse["ellipse"]["axes"][0] / 2), int(proj_ellipse["ellipse"]["axes"][1] / 2)),
                    #            angle=180-int(proj_ellipse["ellipse"]["angle"]),
                    #            arcStart=0,
                    #            arcEnd=360,
                    #            delta=1,
                    #        )
                    
                    
                    # Display 3d model outline in eye window
                    ellipse = pupil3d_datum["projected_sphere"]
                    thickness = 2
                    try:
                        if flip:
                            pts = cv2.ellipse2Poly(
                                center=(int(ellipse["center"][0]), height - int(ellipse["center"][1])),
                                axes=(int(ellipse["axes"][0] / 2), int(ellipse["axes"][1] / 2)),
                                angle=int(ellipse["angle"]),
                                arcStart=0,
                                arcEnd=360,
                                delta=1,
                            )
                        else:
                            pts = cv2.ellipse2Poly(
                                center=(int(ellipse["center"][0]), int(ellipse["center"][1])),
                                axes=(int(ellipse["axes"][0] / 2), int(ellipse["axes"][1] / 2)),
                                angle=int(ellipse["angle"]),
                                arcStart=0,
                                arcEnd=360,
                                delta=1,
                            )
                        #draw_polyline(pts, thickness, RGBA(*rgba))
                    except Exception as e:
                        pts = []
                    debug_frame = np.array(frame)#np.fliplr(frame)
                    if flip:
                        debug_frame = np.flipud(debug_frame)
                    debug_frame = np.ascontiguousarray(debug_frame)
                    cv2.drawContours(debug_frame, np.array([pts]), -1, (0, 255, 0), thickness=2)
                    
                    # Display detected 2d pupil outline in eye window
                    try:
                        if flip:
                            pupilpts = cv2.ellipse2Poly(
                                center=(int(datum_2d["ellipse"]["center"][0]), height - int(datum_2d["ellipse"]["center"][1])),
                                axes=(int(datum_2d["ellipse"]["axes"][0] / 2), int(datum_2d["ellipse"]["axes"][1] / 2)),
                                angle=180-int(datum_2d["ellipse"]["angle"]),
                                arcStart=0,
                                arcEnd=360,
                                delta=1,
                            )
                        else:
                            pupilpts = cv2.ellipse2Poly(
                                center=(int(datum_2d["ellipse"]["center"][0]), int(datum_2d["ellipse"]["center"][1])),
                                axes=(int(datum_2d["ellipse"]["axes"][0] / 2), int(datum_2d["ellipse"]["axes"][1] / 2)),
                                angle=int(datum_2d["ellipse"]["angle"]),
                                arcStart=0,
                                arcEnd=360,
                                delta=1,
                            )
                    except Exception as e:
                        pupilpts = []
                    cv2.drawContours(debug_frame, np.array([pupilpts]), -1, (255, 0, 0), thickness=2)
                    
                    # Display reprojected 3d pupil outline in eye window
                    try:
                        if flip:
                            reproj_pupilpts = cv2.ellipse2Poly(
                                center=(int(proj_ellipse["center"][0]), height - int(proj_ellipse["center"][1])),
                                axes=(int(proj_ellipse["axes"][0] / 2), int(proj_ellipse["axes"][1] / 2)),
                                angle=180-int(proj_ellipse["angle"]),
                                arcStart=0,
                                arcEnd=360,
                                delta=1,
                            )
                        else:
                            reproj_pupilpts = cv2.ellipse2Poly(
                                center=(int(proj_ellipse["center"][0]), int(proj_ellipse["center"][1])),
                                axes=(int(proj_ellipse["axes"][0] / 2), int(proj_ellipse["axes"][1] / 2)),
                                angle=int(proj_ellipse["angle"]),
                                arcStart=0,
                                arcEnd=360,
                                delta=1,
                            )
                        cv2.drawContours(debug_frame, np.array([reproj_pupilpts]), -1, (0, 0, 255), thickness=2)
                    except Exception as e:
                        reproj_pupilpts = []
                    
                    #datum_list["debug_imgs"].append(debug_frame)  # Only needed for annotated world videos -- very memory intensive
                    
                    cv2.imshow("debug", debug_frame)
                    cv2.waitKey(1)
                    #success, frame = vidcap.read()
                bar()
            #if delta_fig_path is not None:
            #    os.makedirs(delta_fig_path+"/"+plugin_name+"/", exist_ok=True)
            #    plt.figure(figsize=(23, 4.8))
            #    #plt.plot(timestamps,centeraz, "-b", label="Pupil X")
            #    #plt.plot(timestamps,centerel, "-r", label="Pupil Y")
            #    plt.plot(np.array(timestamps) - np.min(timestamps), centerdelta, "-b", label="Pupil Center Delta")
            #    plt.legend()
            #    #plt.ylim(0.0, 400.0)
            #    plt.title('2D Ellipse Delta')
            #    plt.xlabel('timestamp (seconds)')
            #    plt.ylabel('delta')
            #    plt.savefig(delta_fig_path+"/"+plugin_name+"/eye"+recording_path[-5]+".png")
            #    plt.clf()
                #plt.show()
    
    
    
    if False:
        plt.plot(list(range(len(ious))), ious, label="IoU over frames")
        plt.title('IoU over time')
        plt.xlabel('frame')
        plt.ylabel('IoU')
        plt.savefig(delta_fig_path+"/"+plugin_name+"/iou_over_frames-eye"+recording_path[-5]+".png")
        plt.show()
        plt.clf()
        
        plt.plot(list(range(len(plconfs))), plconfs, label="PL conf over frames")
        plt.title('PL conf over time')
        plt.xlabel('frame')
        plt.ylabel('PL conf')
        plt.savefig(delta_fig_path+"/"+plugin_name+"/pl_conf_over_frames-eye"+recording_path[-5]+".png")
        plt.show()
        plt.clf()
        
        fig, axs = plt.subplots(2)
        fig.suptitle('Reprojection error over time')
        
        axs[0].plot(list(range(len(reproj_errors_centroid))), reproj_errors_centroid)
        axs[0].set(ylabel='centroid delta reprojection error')
        axs[1].plot(list(range(len(reproj_errors_axes))), reproj_errors_axes)
        axs[1].set(ylabel='axes diff reprojection error')
        #plt.title('Reprojection error (centroid delta) over time')
        plt.xlabel('frame')
        plt.savefig(delta_fig_path+"/"+plugin_name+"/reprojection_error_over_time-eye"+recording_path[-5]+".png")
        plt.show()
        plt.clf()
    
    cv2.destroyAllWindows()
    return datum_list


def get_datum_dicts_from_eyes(file_names, recording_loc, intrinsics, pupil_params, detector_plugin=None, load_2d_pupils=False, skip_3d_detection=False, start_model_timestamp=None, freeze_model_timestamp=None, mapping_method=None):
    recording_dicts = []
    for file_name in file_names:
        eye_id = int(file_name[-5])
        current_intrinsics = intrinsics[eye_id]
        recording_path = recording_loc + "/" + file_name
        if not exists(recording_path):
            logging.error(f"Recording {file_name} does not exist.")
        else:
            logging.debug(f"Recording {file_name} exists!")
            recording_dicts.append(
                pl_detection_on_video(recording_path, fake_gpool(intrinsics[eye_id]), pupil_params, detector_plugin=detector_plugin, delta_fig_path=recording_loc+"/deltas", load_2d_pupils=load_2d_pupils, skip_3d_detection=skip_3d_detection, start_model_timestamp=start_model_timestamp, freeze_model_timestamp=freeze_model_timestamp, mapping_method=mapping_method)
            )
            logging.info(f"Completed detection of recording {file_name}")
    return recording_dicts


def save_datums_to_pldata(datums, save_location, world_file, display_world_video=False):
    import file_methods as fm
    import player_methods as pm

    pupil_data_store = pm.PupilDataCollector()
    for eyeId in range(len(datums)):
        for detector in datums[eyeId]:
            if detector != "debug_imgs":
                for datum in datums[eyeId][detector]:
                    timestamp = datum["timestamp"]
                    pupil_data_store.append(
                        f"pupil.{eyeId}.{detector}", datum, timestamp
                    )  # possibly?
    data = pupil_data_store.as_pupil_data_bisector()
    os.makedirs(save_location, exist_ok=True)
    data.save_to_file(save_location, "offline_pupil")
    session_data = {}
    session_data["detection_status"] = ["complete", "complete"]
    session_data["version"] = 4
    cache_path = os.path.join(save_location, "offline_pupil.meta")
    fm.save_object(session_data, cache_path)
    
    if world_file is not None:
        alpha = 0.75
        beta = 1.0-alpha
        world_cap = cv2.VideoCapture(world_file)
        ret, world_frame = world_cap.read()
        
        world_width  = world_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        world_height  = world_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        world_fps = world_cap.get(cv2.CAP_PROP_FPS)
        world_framecount = world_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        world_duration = world_framecount/world_fps
        
        logging.info(f"Found {world_fps}fps {world_width}x{world_height} world video at {world_file}.")
        eye0_fps = len(datums[0]["2d"])/world_duration
        eye1_fps = len(datums[1]["2d"])/world_duration
        logging.info(f"Found {eye0_fps}fps and {eye1_fps}fps eye videos.")
        world_idx = 0
        eye0_idx = 0
        eye1_idx = 0

        world_timestamps_path = save_location + "/../../world_timestamps.npy"
        eye0_timestamps_path = save_location + "/../../eye0_timestamps.npy"
        eye1_timestamps_path = save_location + "/../../eye1_timestamps.npy"
        eye0_timestamps = np.load(eye0_timestamps_path)
        eye1_timestamps = np.load(eye1_timestamps_path)
        world_timestamps = np.load(world_timestamps_path)

        world_timestamp = world_timestamps[world_idx]
        eye0_timestamp = eye0_timestamps[eye0_idx]
        eye1_timestamp = eye1_timestamps[eye1_idx]

        true_world_fps = np.max([world_framecount, len(datums[0]["2d"]), len(datums[1]["2d"])]) / world_duration

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(save_location, "debug_world.avi"), fourcc, true_world_fps, (int(world_width), int(world_height)))
        cv2.waitKey(1)
        
        if False:  # Don't create a world video -- it's too memory intensive
            logging.info(f"Annotating world video...")
            with alive_bar(int(world_framecount), bar = "filling") as bar:
                while world_cap.isOpened():
                    if not ret:
                        break
                    
                    #world_timestamp = world_idx / world_fps
                    while (eye0_idx < len(datums[0]["2d"]) and eye0_timestamps[eye0_idx] < world_timestamps[world_idx])\
                        or (eye1_idx < len(datums[1]["2d"]) and eye1_timestamps[eye1_idx] < world_timestamps[world_idx]):
                        orig_world_frame = np.copy(world_frame)
                        # Keep repeating this and potentially incrementing the eye timestamps by 1 until it's time to advance to the next World frame
                        eye0_frame = datums[0]["2d"][eye0_idx]
                        eye1_frame = datums[1]["2d"][eye1_idx]
                        eye0_frame_resized = cv2.resize(eye0_frame, (200,200))
                        eye1_frame_resized = cv2.resize(eye1_frame, (200,200))
                        
                        eye0_dst = cv2.addWeighted(eye0_frame_resized, alpha, orig_world_frame[0:200, 0:200, :], beta, 0.0)
                        eye1_dst = cv2.addWeighted(eye1_frame_resized, alpha, orig_world_frame[0:200, int(world_width-200):int(world_width), :], beta, 0.0)
                        orig_world_frame[0:200, 0:200, :] = eye0_dst
                        orig_world_frame[0:200, int(world_width-200):int(world_width), :] = eye1_dst
                        
                        out.write(orig_world_frame)
                        if display_world_video:
                            cv2.imshow('frame',orig_world_frame)
                            cv2.waitKey(1)
                        
                        if eye0_idx < len(datums[0]["2d"]) and eye0_timestamps[eye0_idx] < world_timestamps[world_idx]:
                            eye0_idx += 1
                        if eye1_idx < len(datums[1]["2d"]) and eye1_timestamps[eye1_idx] < world_timestamps[world_idx]:
                            eye1_idx += 1
                        
                    #while eye0_idx < len(datums[0]["debug_imgs"]) and eye0_timestamps[eye0_idx] < world_timestamps[world_idx]:
                    #    eye0_idx += 1
                    #while eye1_idx < len(datums[1]["debug_imgs"]) and eye1_timestamps[eye1_idx] < world_timestamps[world_idx]:
                    #    eye1_idx += 1

                    #eye0_frame = datums[0]["debug_imgs"][eye0_idx]
                    #eye1_frame = datums[1]["debug_imgs"][eye1_idx]
                    #eye0_frame_resized = cv2.resize(eye0_frame, (200,200))
                    #eye1_frame_resized = cv2.resize(eye1_frame, (200,200))
                    
                    #eye0_dst = cv2.addWeighted(eye0_frame_resized, alpha, world_frame[0:200, 0:200, :], beta, 0.0)
                    #eye1_dst = cv2.addWeighted(eye1_frame_resized, alpha, world_frame[0:200, int(world_width-200):int(world_width), :], beta, 0.0)
                    #world_frame[0:200, 0:200, :] = eye0_dst
                    #world_frame[0:200, int(world_width-200):int(world_width), :] = eye1_dst
                    
                    #out.write(world_frame)
                    #cv2.imshow('frame',world_frame)
                    #cv2.waitKey(1)
                    
                    bar()
                    ret, world_frame = world_cap.read()
                    world_idx += 1
        
        world_cap.release()
        out.release()
        try:
            cv2.destroyAllWindows()
        except Exception as _:
            pass
        logging.info(f"Saved video to {os.path.join(save_location, 'debug_world.avi')}")

def perform_pupil_detection(recording_loc, plugin=None,
                            pupil_params=[{"intensity_range": 23,"pupil_size_min": 10,"pupil_size_max": 100}, {"intensity_range": 23,"pupil_size_min": 10,"pupil_size_max": 100}],
                            world_file=None, load_2d_pupils=False, start_model_timestamp=None, freeze_model_timestamp=None, display_world_video=False, mapping_method=None, skip_3d_detection=False):
    eye0_intrinsics_loc = recording_loc + "/eye0.intrinsics"
    eye1_intrinsics_loc = recording_loc + "/eye1.intrinsics"
    scene_cam_intrinsics_loc = recording_loc + "/world.intrinsics"
    
    eye0_intrinsics = load_intrinsics(eye0_intrinsics_loc)
    eye1_intrinsics = load_intrinsics(eye1_intrinsics_loc)
    scene_cam_intrinsics = load_intrinsics(scene_cam_intrinsics_loc, resolution=(640,480))
    intrinsics = [eye0_intrinsics, eye1_intrinsics, scene_cam_intrinsics]
    datums = get_datum_dicts_from_eyes(
        ["eye0.mp4", "eye1.mp4"], recording_loc, intrinsics, pupil_params, detector_plugin=plugin, load_2d_pupils=load_2d_pupils, skip_3d_detection=skip_3d_detection, start_model_timestamp=start_model_timestamp, freeze_model_timestamp=freeze_model_timestamp, mapping_method=mapping_method
    )
    logging.info(f"Completed detection of pupils.")
    
    world_loc = None
    if world_file != None:
        world_loc = recording_loc+"/"+world_file

    if plugin is None:
        save_datums_to_pldata(datums, recording_loc + "/offline_data/vanilla", world_loc, display_world_video=display_world_video)
        logging.info(f"Saved pldata to disk at {recording_loc+f'/offline_data/vanilla'}.")
    else:
        save_datums_to_pldata(datums, recording_loc + f"/offline_data/{plugin.__name__}", world_loc, display_world_video=display_world_video)
        logging.info(f"Saved pldata to disk at {recording_loc+f'/offline_data/{plugin.__name__}'}.")


@click.command()
@click.option(
    "--core_shared_modules_loc",
    required=False,
    type=click.Path(exists=True),
    envvar="CORE_SHARED_MODULES_LOCATION",
)
@click.option(
    "--recording_loc",
    required=True,
    type=click.Path(exists=True),
    envvar="RECORDING_LOCATION",
)
@click.option(
    "--ref_data_loc",
    required=True,
    type=click.Path(exists=True),
    envvar="REF_DATA_LOCATION",
)
def main(core_shared_modules_loc, recording_loc, ref_data_loc):
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("numexpr").setLevel(logging.WARNING)
    logging.getLogger("OpenGL").setLevel(logging.WARNING)

    if core_shared_modules_loc:
        sys.path.append(core_shared_modules_loc)
    else:
        logging.warning("Core source location unknown. Imports might fail.")

    perform_pupil_detection(recording_loc)


if __name__ == "__main__":
    load_dotenv()
    main()
