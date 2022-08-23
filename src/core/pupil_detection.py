import logging
import os
import sys
import types
from os.path import exists

import click
import cv2
import numpy as np
from dotenv import load_dotenv

from core.pipeline import fake_gpool, load_intrinsics
from pye3d.detector_3d import ellipse2dict
from pye3d.geometry.primitives import Circle, Ellipse
from pye3d.geometry.projections import project_circle_into_image_plane

from alive_progress import alive_bar

import matplotlib.pyplot as plt

def pregenerate_eye_model(detector_3d, datums_2d, start_timestamp, stop_timestamp, threshold=None, aspectratio_threshold=None, freeze=True):
    import file_methods as fm
    prev_center = None
    
    for datum_2d in datums_2d:
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
                datum_2d["confidence"] = 0.75
                datum_2d = fm.Serialized_Dict(python_dict=datum_2d)
        
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
        
        # UPDATE MODELS
        if datum_2d["timestamp"] >= start_timestamp and datum_2d["timestamp"] <= stop_timestamp:
            observation = detector_3d._extract_observation(datum_2d)
            detector_3d.update_models(observation)
    if freeze:
        setattr(detector_3d, "is_long_term_model_frozen", True)
    return freeze


def pl_detection_on_video(recording_path, g_pool, pupil_params, detector_plugin=None, debug_window=False, delta_fig_path=None, load_2d_pupils=False, start_model_timestamp=None, freeze_model_timestamp=None):
    IOU_threshold = 0.98
    Delta_threshold = 85.0
    Aspect_ratio_threshold = 0.8
    
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

    vidcap = cv2.VideoCapture(recording_path)
    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    datum_list = {"2d": [], "3d": [], "debug_imgs":[]}
    success, frame = vidcap.read()
    height, width = frame[:, :, 0].shape
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
    """
    pupil_params = [
        {  # eye 0 pupil detection properties
            "intensity_range": 9,
            "pupil_size_min": 10,
            "pupil_size_max": 82
        },
        {  # eye 1 pupil detection properties
            "intensity_range": 9,
            "pupil_size_min": 10,
            "pupil_size_max": 82
        }
    ]
    """
    detector2d = Detector2DPlugin(g_pool=g_pool, properties=pupil_params[id])
    #detector2d_ = detector2d.detector_ritnet_2d
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
    if load_2d_pupils:
        logging.info(f"Loading previously detected pupil datums of {recording_path} ({plugin_name})")
        with alive_bar(int(total_frames), bar = "filling") as bar:
            import player_methods as pm
            recording_loc = recording_path[:-9]
            file_path = recording_loc + "/offline_data/"+plugin_name
            pupil_datums_bisector = pm.PupilDataBisector.load_from_file(file_path, "offline_pupil")
            for loaded_datum in pupil_datums_bisector[recording_path[-5],"2d"]:
                if not success:
                    break
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
                success, frame = vidcap.read()
                bar()
    else:
        logging.info(f"{recording_path} ({plugin_name}) 2D:")
        with alive_bar(int(total_frames), bar = "filling") as bar:
            # Pass 1 - 2D detector
            while success:
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
                success, frame = vidcap.read()
    
    #np.histogram(centerdelta, bins=np.ceil(np.sqrt(height**2 + width**2)) * 2)
    plt.hist(centerdelta, bins=int(np.ceil(np.sqrt(height**2 + width**2)) * 2), histtype='step',edgecolor='r',linewidth=3)
    H, bins = np.histogram(centerdelta, bins=int(np.ceil(np.sqrt(height**2 + width**2)) * 2))
    H+=np.histogram(centerdelta, bins=int(np.ceil(np.sqrt(height**2 + width**2)) * 2))[0]

    plt.bar(bins[:-1],H,width=1)
    
    print("1:",np.percentile(centerdelta, 1.0))
    print("2:",np.percentile(centerdelta, 2.0))
    print("3:",np.percentile(centerdelta, 3.0))
    print("4:",np.percentile(centerdelta, 4.0))
    print("5:",np.percentile(centerdelta, 5.0))
    print("6:",np.percentile(centerdelta, 6.0))
    print("7:",np.percentile(centerdelta, 7.0))
    print("8:",np.percentile(centerdelta, 8.0))
    print("9:",np.percentile(centerdelta, 9.0))
    print("10:",np.percentile(centerdelta, 10.0))
    print("50:",np.percentile(centerdelta, 50.0))
    print("80:",np.percentile(centerdelta, 80.0))
    print("81:",np.percentile(centerdelta, 81.0))
    print("82:",np.percentile(centerdelta, 82.0))
    print("83:",np.percentile(centerdelta, 83.0))
    print("84:",np.percentile(centerdelta, 84.0))
    print("85:",np.percentile(centerdelta, 85.0))
    print("86:",np.percentile(centerdelta, 86.0))
    print("87:",np.percentile(centerdelta, 87.0))
    print("88:",np.percentile(centerdelta, 88.0))
    print("89:",np.percentile(centerdelta, 89.0))
    print("90:",np.percentile(centerdelta, 90.0))
    print("91:",np.percentile(centerdelta, 91.0))
    print("92:",np.percentile(centerdelta, 92.0))
    print("93:",np.percentile(centerdelta, 93.0))
    print("94:",np.percentile(centerdelta, 94.0))
    print("95:",np.percentile(centerdelta, 95.0))
    print("96:",np.percentile(centerdelta, 96.0))
    print("97:",np.percentile(centerdelta, 97.0))
    print("98:",np.percentile(centerdelta, 98.0))
    print("99:",np.percentile(centerdelta, 99.0))
    print("100:",np.percentile(centerdelta, 100.0))
    print()

    plt.title("Pupil Centroid Celta")
    #plt.show()
    plt.clf()
    if Delta_threshold is None:
        threshold = None
    else:
        threshold = np.percentile(centerdelta, Delta_threshold)  # x% least moving pupil centroids
    
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
    
    # Generate model based on entire video
    #start_model_timestamp = 0.0
    #from operator import itemgetter
    #freeze_model_timestamp = max(datum_list["2d"], key=itemgetter("timestamp"))["timestamp"]

    has_frozen = False
    #import copy
    #sys.setrecursionlimit(10000)
    #temp_list = []
    #for datum in datum_list["2d"]:
    #    temp_list.append(datum["confidence"])
    
    sync_updates_PL = False
    start_model_timestamp = None
    freeze_model_timestamp = None
    if not sync_updates_PL and (start_model_timestamp is not None and freeze_model_timestamp is not None):
        logging.info(f"Pregenerating 3D eye model...")
        has_frozen = pregenerate_eye_model(detector3d.pupil_detector, datum_list["2d"], start_model_timestamp, freeze_model_timestamp, threshold, aspectratio_threshold=Aspect_ratio_threshold, freeze=True)
    
    #for i in range(len(datum_list["2d"])):
    #    datum_list["2d"][i]["confidence"] = temp_list[i]
    #datum_list["2d"] = temp_list
    
    vidcap = cv2.VideoCapture(recording_path)
    success, frame = vidcap.read()
    logging.info(f"{recording_path} ({plugin_name}) 3D:")
    prev_center = None
    
    if sync_updates_PL:
        new_detector2d = VanillaDetector2DPlugin(g_pool=g_pool, properties=pupil_params[id])
    
    with alive_bar(int(total_frames), bar = "filling") as bar:
        # Pass 2 - 3D detector
        for idx, datum_2d in enumerate(datum_list["2d"]):
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
            
            # VELOCITY THRESHOLD
            if threshold is not None and delta > threshold:
                datum_2d = datum_2d._deep_copy_dict()
                datum_2d["confidence"] = 0.75
                datum_2d = fm.Serialized_Dict(python_dict=datum_2d)
            
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
            
            datum_list["debug_imgs"].append(debug_frame)
            if debug_window:
                cv2.imshow("debug", debug_frame)
                cv2.waitKey(1)
            success, frame = vidcap.read()
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


def get_datum_dicts_from_eyes(file_names, recording_loc, intrinsics, pupil_params, detector_plugin=None, load_2d_pupils=False, start_model_timestamp=None, freeze_model_timestamp=None):
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
                pl_detection_on_video(recording_path, fake_gpool(intrinsics[eye_id]), pupil_params, detector_plugin=detector_plugin, delta_fig_path=recording_loc+"/deltas", load_2d_pupils=load_2d_pupils, start_model_timestamp=start_model_timestamp, freeze_model_timestamp=freeze_model_timestamp)
            )
            logging.info(f"Completed detection of recording {file_name}")
    return recording_dicts


def save_datums_to_pldata(datums, save_location, world_file):
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
        eye0_fps = len(datums[0]["debug_imgs"])/world_duration
        eye1_fps = len(datums[1]["debug_imgs"])/world_duration
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

        true_world_fps = np.max([world_framecount, len(datums[0]["debug_imgs"]), len(datums[1]["debug_imgs"])]) / world_duration

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(save_location, "debug_world.avi"), fourcc, true_world_fps, (int(world_width), int(world_height)))
        cv2.waitKey(1)
        
        while world_cap.isOpened():
            if not ret:
                break
            
            #world_timestamp = world_idx / world_fps
            while (eye0_idx < len(datums[0]["debug_imgs"]) and eye0_timestamps[eye0_idx] < world_timestamps[world_idx])\
                or (eye1_idx < len(datums[1]["debug_imgs"]) and eye1_timestamps[eye1_idx] < world_timestamps[world_idx]):
                orig_world_frame = np.copy(world_frame)
                # Keep repeating this and potentially incrementing the eye timestamps by 1 until it's time to advance to the next World frame
                eye0_frame = datums[0]["debug_imgs"][eye0_idx]
                eye1_frame = datums[1]["debug_imgs"][eye1_idx]
                eye0_frame_resized = cv2.resize(eye0_frame, (200,200))
                eye1_frame_resized = cv2.resize(eye1_frame, (200,200))
                
                eye0_dst = cv2.addWeighted(eye0_frame_resized, alpha, orig_world_frame[0:200, 0:200, :], beta, 0.0)
                eye1_dst = cv2.addWeighted(eye1_frame_resized, alpha, orig_world_frame[0:200, int(world_width-200):int(world_width), :], beta, 0.0)
                orig_world_frame[0:200, 0:200, :] = eye0_dst
                orig_world_frame[0:200, int(world_width-200):int(world_width), :] = eye1_dst
                
                out.write(orig_world_frame)
                cv2.imshow('frame',orig_world_frame)
                cv2.waitKey(1)
                
                if eye0_idx < len(datums[0]["debug_imgs"]) and eye0_timestamps[eye0_idx] < world_timestamps[world_idx]:
                    eye0_idx += 1
                if eye1_idx < len(datums[1]["debug_imgs"]) and eye1_timestamps[eye1_idx] < world_timestamps[world_idx]:
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
            
            ret, world_frame = world_cap.read()
            world_idx += 1
        
        world_cap.release()
        out.release()
        cv2.destroyAllWindows()
        logging.info(f"Saved video to {os.path.join(save_location, 'debug_world.avi')}")

def perform_pupil_detection(recording_loc, plugin=None,
                            pupil_params=[{"intensity_range": 23,"pupil_size_min": 10,"pupil_size_max": 100}, {"intensity_range": 23,"pupil_size_min": 10,"pupil_size_max": 100}],
                            world_file=None, load_2d_pupils=False, start_model_timestamp=None, freeze_model_timestamp=None):
    eye0_intrinsics_loc = recording_loc + "/eye0.intrinsics"
    eye1_intrinsics_loc = recording_loc + "/eye1.intrinsics"
    scene_cam_intrinsics_loc = recording_loc + "/world.intrinsics"
    
    eye0_intrinsics = load_intrinsics(eye0_intrinsics_loc)
    eye1_intrinsics = load_intrinsics(eye1_intrinsics_loc)
    scene_cam_intrinsics = load_intrinsics(scene_cam_intrinsics_loc, resolution=(640,480))
    intrinsics = [eye0_intrinsics, eye1_intrinsics, scene_cam_intrinsics]
    datums = get_datum_dicts_from_eyes(
        ["eye0.mp4", "eye1.mp4"], recording_loc, intrinsics, pupil_params, detector_plugin=plugin, load_2d_pupils=load_2d_pupils, start_model_timestamp=start_model_timestamp, freeze_model_timestamp=freeze_model_timestamp
    )
    logging.info(f"Completed detection of pupils.")
    
    world_loc = None
    if world_file != None:
        world_loc = recording_loc+"/"+world_file
    
    if plugin is None:
        save_datums_to_pldata(datums, recording_loc + "/offline_data/vanilla", world_loc)
        logging.info(f"Saved pldata to disk at {recording_loc+f'/offline_data/vanilla'}.")
    else:
        save_datums_to_pldata(datums, recording_loc + f"/offline_data/{plugin.__name__}", world_loc)
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
