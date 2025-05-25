from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from hercules.aeva import read_aeva_bin_for_foxglove
from hercules.transforms import get_stereo_lidar_transforms_msgs
from hercules.camera import read_stereo_image_for_foxglove
from utils.files import write_to_mcap, get_chunks

def aeva_binfile_to_mcap(data_folder_path:Path):
    """Convert Aeva binary files to MCAP format.
    This function processes Aeva LiDAR binary files from a specified data folder, 
    converts them to Foxglove-compatible MCAP format, and generates associated 
    transform messages. The files are processed in chunks, and for each chunk, 
    two MCAP files are created: one containing the LiDAR data and another 
    containing transform data.
    Args:
        data_folder_path (Path): Path to the data folder containing Aeva binary files.
            The function expects these files to be located in "Avea_data/LiDAR/Aeva/*.bin".
            The binary files should be named with numerical values that can be sorted.
    Returns:
        None: The function writes output to MCAP files in a "mcaps" subdirectory of the 
        input folder but does not return any values.
    Note:
        - The function creates output directories if they don't exist.
        - Two MCAP files are generated for each chunk of binary files:
          1. A file containing LiDAR data with the topic "hercules/aeva"
          2. A file containing transform data with the topic "hercules/transform"
    """

    transformantion_matrix_lcam = [[0.022023532009715, -0.999669977147658, 0.013225007652863, 0.254288466421238],
                            [-0.049191798363281, -0.014295738779819, -0.998687037477971, -0.108349681409513],
                            [0.998546509188032, 0.021344054027771, -0.049490406606293,  -0.136209414128968]
                            ]
    
    transformantion_matrix_rcam = [[0.029707602336548, -0.999557708040526, -0.001358918755325, -0.259694516606996],
                                   [-0.033598758951609, 0.0003601721560598965, -0.999435337414547, -0.154192126023634], 
                                   [0.998993784645493, 0.029736485548700, -0.033573198639491, -0.136373949928340]
                            ]

    aeva_binfiles = [Path(path) for path in data_folder_path.glob("Avea_data/LiDAR/Aeva/*.bin")]
    aeva_binfiles.sort(key=lambda x: int(x.stem))
    chunks = get_chunks(aeva_binfiles)
    
    for chunk in chunks:
        foxglove_msgs = [(read_aeva_bin_for_foxglove(bin_path, timestamp=int(bin_path.stem)), int(bin_path.stem)) for bin_path  in chunk]
        
        output_aeva_mcap_path = Path(f"{data_folder_path}/mcaps/{chunk[0].stem}_to_{chunk[-1].stem}_aeva.mcap")
        if not output_aeva_mcap_path.parent.exists():
            output_aeva_mcap_path.parent.mkdir(parents=True, exist_ok=True)

        write_to_mcap(foxglove_msgs, output_aeva_mcap_path, topic="hercules/aeva")
        print(f"Converted Aeva data to {output_aeva_mcap_path}")

        # Create all transform messages for each timestamp in the chunk
        transform_msgs = []
        for bin_path in chunk:
            timestamp = int(bin_path.stem)
            msgs = get_stereo_lidar_transforms_msgs(timestamp, 
                                                transformantion_matrix_lcam=transformantion_matrix_lcam, 
                                                transformantion_matrix_rcam=transformantion_matrix_rcam)
            transform_msgs.extend(msgs)

        output_transform_mcap_path = Path(f"{data_folder_path}/mcaps/{chunk[0].stem}_to_{chunk[-1].stem}_aeva_transforms.mcap")
        if not output_transform_mcap_path.parent.exists():
            output_transform_mcap_path.parent.mkdir(parents=True, exist_ok=True)

        write_to_mcap(transform_msgs, output_transform_mcap_path, topic="hercules/transform")
        print(f"Converted Aeva transform data to {output_transform_mcap_path}")


def stereo_image_to_mcap(data_folder_path:Path):
    """Convert stereo images to MCAP format.
    This function processes stereo images from a specified data folder, 
    converts them to Foxglove-compatible MCAP format, and generates associated 
    transform messages. The files are processed in chunks, and for each chunk, 
    two MCAP files are created: one containing the stereo image data and another 
    containing transform data. The function expects the stereo images to be located 
    in "Avea_data/Image/stereo_left/*.png" and "Avea_data/Image/stereo_right/*.png".
    
    Args:
        data_folder_path (Path): Path to the data folder containing stereo images.
            The function expects these files to be located in "Avea_data/Image/stereo_left/*.png" and "Avea_data/Image/stereo_right/*.png".
            The image files should be named with numerical values that can be sorted.
    
    Returns:
        None: The function writes output to MCAP files in a "mcaps" subdirectory of the 
        input folder but does not return any values.
    
    Note:
        - The function creates output directories if they don't exist.
        - Two MCAP files are generated for each chunk of stereo images:
          1. A file containing stereo image data with the topic "hercules/stereo"
    """
    
    image_file_paths = [Path(path) for path in data_folder_path.glob("Image/stereo_left/*.png")]
    image_file_paths.sort(key=lambda x: int(x.stem))
    
    aeva_mcaps = list(Path(data_folder_path).glob("mcaps/*_aeva.mcap"))
    aeva_mcaps.sort(key=lambda x: int(x.stem.split("_")[0]))
    chunks = []
    for mcaps in aeva_mcaps:
        mcaps_start_stamp = mcaps.stem.split("_")[0]
        mcaps_end_stamp = mcaps.stem.split("_")[-1]
        image_file_paths = [image_path for image_path in image_file_paths if mcaps_start_stamp <= image_path.stem <= mcaps_end_stamp]
        chunks.append(image_file_paths)

    distortion_param = [-0.007761339207475, 0.0009533009641353071]
    intrinsic_param = [490.2369426861542, 0.0, 735.9762455083395, 0.0, 489.9420108174773, 582.3039081302234, 0.0, 0.0, 1.0]
     

    for chunk in chunks:
        foxglove_msgs = [(read_stereo_image_for_foxglove(image_path, distortion_param=distortion_param, intrinsic_param=intrinsic_param), int(image_path.stem)) for image_path in tqdm(chunk, desc="Generating stereo messages for MCAP")]

        output_mcap_path = Path(f"{data_folder_path}/mcaps/{chunk[0].stem}_to_{chunk[-1].stem}_stereo.mcap")
        if not output_mcap_path.parent.exists():
            output_mcap_path.parent.mkdir(parents=True, exist_ok=True)

        write_to_mcap(foxglove_msgs, output_mcap_path, topic="hercules/stereo")
        print(f"Converted Stereo data to {output_mcap_path}")

if __name__ == '__main__':
    data_folder_path = Path("data")
    #aeva_binfile_to_mcap(data_folder_path)
    stereo_image_to_mcap(data_folder_path)
    print("Conversion completed successfully.")