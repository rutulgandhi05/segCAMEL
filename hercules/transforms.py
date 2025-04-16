from pathlib import Path
from datetime import datetime
from misc.mcap_writer import write_to_mcap
from mcap_protobuf.reader import read_protobuf_messages as PBReader
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform as PB_FrameTransform
from foxglove_schemas_protobuf.Vector3_pb2 import Vector3 as PB_Vector3
from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion as PB_Quaternion
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

def get_stereo_lidar_transforms_msgs(timestamp: int, transformantion_matrix_lcam=None, transformantion_matrix_rcam=None):
    stamp = datetime.fromtimestamp(timestamp/1000000000)  
    transforms = [] 

    if transformantion_matrix_lcam:
        rotation_matrix_lcam = [transformantion_matrix_lcam[0][:3],
                                transformantion_matrix_lcam[1][:3],
                                transformantion_matrix_lcam[2][:3],
                                ]
        R_caml = R.from_matrix(rotation_matrix_lcam[:3][:3]).as_quat()
        translation_lcam = [transformantion_matrix_lcam[0][3], transformantion_matrix_lcam[1][3], transformantion_matrix_lcam[2][3]]

        transform_l = PB_FrameTransform(translation = PB_Vector3(x=translation_lcam[0], y=translation_lcam[1], z=translation_lcam[2]),
                                        rotation = PB_Quaternion(x=R_caml[0], y=R_caml[1], z=R_caml[2], w=R_caml[3]))
        transform_l.timestamp = stamp
        transform_l.parent_frame_id = "stereo"
        transform_l.child_frame_id = "aeva"
        transforms.append([transform_l, timestamp])
    
    if transformantion_matrix_rcam:
        rotation_matrix_rcam = [transformantion_matrix_rcam[0][:3],
                                transformantion_matrix_rcam[1][:3],
                                transformantion_matrix_rcam[2][:3],
                                ]
        R_camr = R.from_matrix(rotation_matrix_rcam[:3][:3]).as_quat()
        translation_rcam = [transformantion_matrix_rcam[0][3], transformantion_matrix_rcam[1][3], transformantion_matrix_rcam[2][3]]

        transform_r = PB_FrameTransform(translation = PB_Vector3(x=translation_rcam[0], y=translation_rcam[1], z=translation_rcam[2]), 
                                        rotation = PB_Quaternion(x=R_camr[0], y=R_camr[1], z=R_camr[2], w=R_camr[3]) )
        transform_r.timestamp = stamp
        transform_r.parent_frame_id = "stereo"
        transform_r.child_frame_id = "aeva"
        transforms.append([transform_r, timestamp])

    return transforms

if __name__ == '__main__':
    # Example usage

    transformantion_matrix_lcam = [[0.022023532009715, -0.999669977147658, 0.013225007652863, 0.254288466421238],
                            [-0.049191798363281, -0.014295738779819, -0.998687037477971, -0.108349681409513],
                            [0.998546509188032, 0.021344054027771, -0.049490406606293,  -0.136209414128968]
                            ]
    
    transformantion_matrix_rcam = [[0.029707602336548, -0.999557708040526, -0.001358918755325, -0.259694516606996],
                                   [-0.033598758951609, 0.0003601721560598965, -0.999435337414547, -0.154192126023634], 
                                   [0.998993784645493, 0.029736485548700, -0.033573198639491, -0.136373949928340]
                            ]

    print(get_stereo_lidar_transforms_msgs(1724813019217024258, transformantion_matrix_lcam, transformantion_matrix_rcam))