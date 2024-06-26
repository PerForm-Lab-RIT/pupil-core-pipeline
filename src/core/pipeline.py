import logging
import os
import pathlib
import sys
import time
import types
import typing as T
from math import floor

import click
from dotenv import load_dotenv

from alive_progress import alive_bar


def save_gaze_data(gaze, gaze_ts, recording_loc, plugin=None, export=True):
    import file_methods as fm
    
    if plugin is None:
        directory = os.path.join(recording_loc, "pipeline-gaze-mappings", "vanilla")
        export_directory = os.path.join(recording_loc, "Exports", "vanilla")
    else:
        directory = os.path.join(recording_loc, "pipeline-gaze-mappings", plugin.__name__)
        export_directory = os.path.join(recording_loc, "Exports", plugin.__name__)
    os.makedirs(directory, exist_ok=True)
    file_name = "pipeline"  # self._gaze_mapping_file_name(gaze_mapper)
    with fm.PLData_Writer(directory, file_name) as writer:
        for gaze_ts_uz, gaze_uz in zip(gaze_ts, gaze):
            writer.append_serialized(
                gaze_ts_uz, topic="gaze", datum_serialized=gaze_uz.serialized
            )
    logging.info(f"Gaze data saved to {directory}.")

    if export:
        import player_methods as pm
        import csv_utils
        from raw_data_exporter import Gaze_Positions_Exporter

        class My_Gaze_Positions_Exporter(Gaze_Positions_Exporter):
            @classmethod
            def csv_export_labels(cls) -> T.Tuple[csv_utils.CSV_EXPORT_LABEL_TYPE, ...]:
                return Gaze_Positions_Exporter.csv_export_labels() + ('pupil_confidence0', 'pupil_confidence1', 'pupil_centroid0_x',
                                                            'pupil_centroid0_y', 'pupil_centroid1_x', 'pupil_centroid1_y',
                                                            'deprojected_norm_pos_x', 'deprojected_norm_pos_y', 'deprojected_norm_pos_z',
                                                            'deprojected_norm_pos0_x', 'deprojected_norm_pos0_y', 'deprojected_norm_pos0_z',
                                                            'deprojected_norm_pos1_x', 'deprojected_norm_pos1_y', 'deprojected_norm_pos1_z',)
            
            @classmethod
            def dict_export(
                cls, raw_value: csv_utils.CSV_EXPORT_RAW_TYPE, world_index: int
            ) -> dict:
                res = Gaze_Positions_Exporter.dict_export(raw_value, world_index)
                res['pupil_confidence0'] = 0.0
                res['pupil_confidence1'] = 0.0
                res['pupil_centroid0_x'] = 0.0
                res['pupil_centroid0_y'] = 0.0
                res['pupil_centroid1_x'] = 0.0
                res['pupil_centroid1_y'] = 0.0
                if raw_value.get("base_data", None) is not None:
                    for v in raw_value['base_data']:
                        if v['id'] == 0:
                            res['pupil_confidence0'] = v['confidence']
                            res['pupil_centroid0_x'] = v['norm_pos'][0]#v['center'][0]
                            res['pupil_centroid0_y'] = v['norm_pos'][1]#v['center'][1]
                        elif v['id'] == 1:
                            res['pupil_confidence1'] = v['confidence']
                            res['pupil_centroid1_x'] = v['norm_pos'][0]#v['center'][0]
                            res['pupil_centroid1_y'] = v['norm_pos'][1]#v['center'][1]
                if raw_value.get('deprojected_norm_pos', None) is not None:
                    res['deprojected_norm_pos_x'] = raw_value['deprojected_norm_pos'][0]
                    res['deprojected_norm_pos_y'] = raw_value['deprojected_norm_pos'][1]
                    res['deprojected_norm_pos_z'] = raw_value['deprojected_norm_pos'][2]
                
                if raw_value.get('deprojected_norm_pos0', None) is not None:
                    res['deprojected_norm_pos0_x'] = raw_value['deprojected_norm_pos0'][0]
                    res['deprojected_norm_pos0_y'] = raw_value['deprojected_norm_pos0'][1]
                    res['deprojected_norm_pos0_z'] = raw_value['deprojected_norm_pos0'][2]
                if raw_value.get('deprojected_norm_pos1', None) is not None:
                    res['deprojected_norm_pos1_x'] = raw_value['deprojected_norm_pos1'][0]
                    res['deprojected_norm_pos1_y'] = raw_value['deprojected_norm_pos1'][1]
                    res['deprojected_norm_pos1_z'] = raw_value['deprojected_norm_pos1'][2]
                return res
        
        os.makedirs(export_directory, exist_ok=True)
        gaze_bisector = pm.Bisector(gaze, gaze_ts)
        My_Gaze_Positions_Exporter = My_Gaze_Positions_Exporter()
        My_Gaze_Positions_Exporter.csv_export_write(
            positions_bisector=gaze_bisector,
            timestamps=gaze_ts,
            export_window=[
                gaze_bisector.data_ts[0] - 1,
                gaze_bisector.data_ts[len(gaze_bisector.data_ts) - 1] + 1,
            ],
            export_dir=export_directory,
        )
        logging.info(f"Gaze data exported to {export_directory}.")


def map_pupil_data(gazer, pupil_data, rec_loc, bar_enabled=True):
    import file_methods as fm
    import numpy as np
    from methods import denormalize

    scene_cam_intrinsics = load_intrinsics(rec_loc+"/world.intrinsics", resolution=(640,480))

    logging.info("Mapping pupil data to gaze data.")
    gaze = []
    gaze_ts = []

    first_ts = pupil_data[0]["timestamp"]
    last_ts = pupil_data[-1]["timestamp"]
    ts_span = last_ts - first_ts
    curr_ts = first_ts

    prev_prog = 0.0
    if bar_enabled:
        with alive_bar(int(len(pupil_data)), bar = "filling") as bar:
            for gaze_datum in gazer.map_pupil_to_gaze(pupil_data):
                curr_ts = max(curr_ts, gaze_datum["timestamp"])
                progress = (curr_ts - first_ts) / ts_span
                #if floor(progress * 100) != floor(prev_prog * 100):
                #    logging.info(f"Gaze Mapping Progress: {floor(progress*100)}%")
                bar()
                prev_prog = progress
                # result = (curr_ts, fm.Serialized_Dict(gaze_datum))
                
                #deprojected = scene_cam_intrinsics.unprojectPoints(np.array([gaze_datum["norm_pos"]]), normalize=True)
                deprojected = scene_cam_intrinsics.unprojectPoints(np.array([denormalize(gaze_datum["norm_pos"], size=(640, 480))]), normalize=True)
                gaze_datum['deprojected_norm_pos'] = deprojected[0].tolist()
                
                try:
                    deprojected_0 = scene_cam_intrinsics.unprojectPoints(np.array([denormalize(gaze_datum["right_norm_pos"], size=(640, 480))]), normalize=True)
                    gaze_datum['deprojected_norm_pos0'] = deprojected_0[0].tolist()
                except KeyError as _:
                    # Modified 2D gazer was not used
                    pass
                except TypeError as _:
                    # Modified 2D gazer was used, but this eye was not detected for this frame
                    pass
                
                try:
                    deprojected_1 = scene_cam_intrinsics.unprojectPoints(np.array([denormalize(gaze_datum["left_norm_pos"], size=(640, 480))]), normalize=True)
                    gaze_datum['deprojected_norm_pos1'] = deprojected_1[0].tolist()
                except KeyError as _:
                    # Modified 2D gazer was not used
                    pass
                except TypeError as _:
                    # Modified 2D gazer was used, but this eye was not detected for this frame
                    pass
                
                gaze.append(fm.Serialized_Dict(gaze_datum))
                gaze_ts.append(curr_ts)
    else:
        for gaze_datum in gazer.map_pupil_to_gaze(pupil_data):
            curr_ts = max(curr_ts, gaze_datum["timestamp"])
            progress = (curr_ts - first_ts) / ts_span
            #if floor(progress * 100) != floor(prev_prog * 100):
            #    logging.info(f"Gaze Mapping Progress: {floor(progress*100)}%")
            prev_prog = progress
            # result = (curr_ts, fm.Serialized_Dict(gaze_datum))

            deprojected = scene_cam_intrinsics.unprojectPoints(np.array([denormalize(gaze_datum["norm_pos"], size=(640, 480))]), normalize=True)
            gaze_datum['deprojected_norm_pos'] = deprojected[0].tolist()
            gaze.append(fm.Serialized_Dict(gaze_datum))
            gaze_ts.append(curr_ts)

    count_2 = 0
    count_1 = 0
    count_0 = 0
    count_other = 0
    for g in gaze:
        if g["topic"] == 'gaze.2d.0.':
            count_0 += 1
        elif g["topic"] == 'gaze.2d.1.':
            count_1 += 1
        elif g["topic"] == 'gaze.2d.01.':
            count_2 += 1
        else:
            count_other += 1

    logging.info("Pupil data mapped to gaze data.")
    return gaze, gaze_ts


def calibrate_and_validate(
    ref_loc, pupil_loc, scene_cam_intrinsics_loc, mapping_method, realtime_ref_loc=None, min_calibration_confidence=0.0
):
    if realtime_ref_loc is not None:
        pupil = load_pupil_data(pupil_loc)
        logging.debug(f"Loaded {len(pupil.data)} pupil positions")
        realtime_ref_data = load_realtime_ref_data(realtime_ref_loc)
        logging.debug(f"Loaded {len(realtime_ref_data)} reference locations")
        return calibrate_and_validate_realtime(realtime_ref_data, pupil, scene_cam_intrinsics_loc, mapping_method, min_calibration_confidence=min_calibration_confidence)

    ref_data = None
    ref_data = load_ref_data(ref_loc)
    logging.debug(f"Loaded {len(ref_data)} reference locations")
    realtime_ref_data = None

    pupil = load_pupil_data(pupil_loc)
    logging.debug(f"Loaded {len(pupil.data)} pupil positions")
    scene_cam_intrinsics = load_intrinsics(scene_cam_intrinsics_loc, resolution=(640,480))
    logging.debug(f"Loaded scene camera intrinsics: {scene_cam_intrinsics}")
    gazer = fit_gazer(mapping_method, ref_data, pupil.data, scene_cam_intrinsics, realtime_ref=realtime_ref_data, min_calibration_confidence=min_calibration_confidence)
    return gazer, pupil.data

def calibrate_and_validate_realtime(

    realtime_ref_data, pupil, scene_cam_intrinsics_loc, mapping_method, min_calibration_confidence=0.0
):
    ref_data = None
    scene_cam_intrinsics = load_intrinsics(scene_cam_intrinsics_loc, resolution=(640,480))
    logging.debug(f"Loaded scene camera intrinsics: {scene_cam_intrinsics}")
    gazer = fit_gazer(mapping_method, ref_data, pupil.data, scene_cam_intrinsics, realtime_ref=realtime_ref_data, min_calibration_confidence=min_calibration_confidence)
    return gazer, pupil.data

def load_ref_data(ref_loc):
    import file_methods as fm

    ref = fm.load_object(ref_loc)
    assert ref["version"] == 1, "unexpected reference data format"
    return [{"screen_pos": r[0], "timestamp": r[2]} for r in ref["data"]]

def load_realtime_ref_data(ref_loc):
    import file_methods as fm
    res = []
    notifications = fm.load_pldata_file(ref_loc[:ref_loc.rfind('/')], "notify")
    for topic, data in zip(notifications.topics, notifications.data):
        if data['subject'] == 'calibration.add_ref_data' or data['subject'] == 'notify.calibration.add_ref_data':
            res = res + [{"screen_pos": r["mm_pos"], "timestamp": r["timestamp"]} for r in data["ref_data"]]
    return res
    
def get_first_realtime_ref_data_timestamp(ref_loc):
    ref_data = load_realtime_ref_data(ref_loc)
    first_timestamp = None
    for ref in ref_data:
        if first_timestamp is None or ref["timestamp"] < first_timestamp:
            first_timestamp = ref["timestamp"]
    return first_timestamp
    
def get_first_ref_data_timestamp(ref_loc):
    ref_data = load_ref_data(ref_loc)
    first_timestamp = None
    for ref in ref_data:
        if first_timestamp is None or ref["timestamp"] < first_timestamp:
            first_timestamp = ref["timestamp"]
    return first_timestamp
    
def get_last_realtime_ref_data_timestamp(ref_loc):
    ref_data = load_realtime_ref_data(ref_loc)
    first_timestamp = None
    for ref in ref_data:
        if first_timestamp is None or ref["timestamp"] > first_timestamp:
            first_timestamp = ref["timestamp"]
    return first_timestamp
    
def get_last_ref_data_timestamp(ref_loc):
    ref_data = load_ref_data(ref_loc)
    first_timestamp = None
    for ref in ref_data:
        if first_timestamp is None or ref["timestamp"] > first_timestamp:
            first_timestamp = ref["timestamp"]
    return first_timestamp

def load_pupil_data(pupil_loc):
    import file_methods as fm

    pupil_loc = pathlib.Path(pupil_loc)
    pupil = fm.load_pldata_file(pupil_loc.parent, pupil_loc.stem)
    return pupil


def load_intrinsics(intrinsics_loc, resolution=None):#(640, 480)):
    import camera_models as cm

    intrinsics_loc = pathlib.Path(intrinsics_loc)
    from file_methods import load_object
    import ast
    
    intrinsics_dict = load_object(intrinsics_loc, allow_legacy=False)
    
    if resolution is None:
        for key in intrinsics_dict.keys():
            if key != 'version':
                res = ast.literal_eval(key)
                if type(res) == type((1,2)):
                    resolution = res
                    break
    
    return cm.Camera_Model.from_file(
        intrinsics_loc.parent, intrinsics_loc.stem, resolution
    )


def available_mapping_methods():
    import gaze_mapping

    return {
        gazer.label: gazer
        for gazer in gaze_mapping.user_selectable_gazer_classes_posthoc()
    }


def fit_gazer(mapping_method, ref_data, pupil_data, scene_cam_intrinsics, realtime_ref=None, min_calibration_confidence=0.0):
    return mapping_method(
        fake_gpool(scene_cam_intrinsics, realtime_ref=realtime_ref, min_calibration_confidence=min_calibration_confidence),
        calib_data={"ref_list": ref_data, "pupil_list": pupil_data},
        posthoc_calib=(realtime_ref is not None),
    )


def fake_gpool(scene_cam_intrinsics, app="pipeline", min_calibration_confidence=0.0, realtime_ref=None):
    g_pool = types.SimpleNamespace()
    g_pool.capture = types.SimpleNamespace()
    g_pool.capture.intrinsics = scene_cam_intrinsics
    g_pool.capture.frame_size = scene_cam_intrinsics.resolution
    g_pool.get_timestamp = time.perf_counter
    g_pool.app = app
    g_pool.min_calibration_confidence = min_calibration_confidence
    g_pool.realtime_ref = realtime_ref
    return g_pool


def patch_plugin_notify_all(plugin_class):
    def log_notification(self, notification):
        """Patches Plugin.notify_all of gazer class"""
        logging.info(f"Notification: {notification['subject']} ({notification.keys()})")

    logging.debug(f"Patching {plugin_class.notify_all}")
    plugin_class.notify_all = log_notification


@click.command()
@click.option("--skip_pupil_detection", is_flag=True)
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
def main(skip_pupil_detection, core_shared_modules_loc, recording_loc, ref_data_loc):
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("numexpr").setLevel(logging.WARNING)
    logging.getLogger("OpenGL").setLevel(logging.WARNING)

    if core_shared_modules_loc:
        sys.path.append(core_shared_modules_loc)
    else:
        logging.warning("Core source location unknown. Imports might fail.")

    mapping_methods_by_label = available_mapping_methods()
    mapping_method_label = click.prompt(
        "Choose gaze mapping method",
        type=click.Choice(mapping_methods_by_label.keys(), case_sensitive=True),
    )
    mapping_method = mapping_methods_by_label[mapping_method_label]
    patch_plugin_notify_all(mapping_method)

    if not skip_pupil_detection:
        from core.pupil_detection import perform_pupil_detection

        logging.info("Performing pupil detection on eye videos. This may take a while.")
        perform_pupil_detection(recording_loc)
        logging.info("Pupil detection complete.")

    pupil_data_loc = recording_loc + "/offline_data/offline_pupil.pldata"
    intrinsics_loc = recording_loc + "/world.intrinsics"
    calibrated_gazer, pupil_data = calibrate_and_validate(
        ref_data_loc, pupil_data_loc, intrinsics_loc, mapping_method
    )
    gaze, gaze_ts = map_pupil_data(calibrated_gazer, pupil_data)
    save_gaze_data(gaze, gaze_ts, recording_loc)


if __name__ == "__main__":
    load_dotenv()
    main()
