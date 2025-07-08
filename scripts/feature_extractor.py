from pathlib import Path
from PIL import Image
from dinotool.utils import frame_visualizer
from dinotool.data import TransformFactory,FrameData, LocalFeatures
from dinotool.model import load_model, DinoFeatureExtractor, PCAModule
import tempfile
from utils.visualization import create_video_from_frames
from tqdm import tqdm

class Extractor:
    def __init__(self, dino_model: str ="dinov2_vits14_reg", clip_model: str ="clip_vit_b32"):
        """
        Initializes the Extractor class for feature extraction using Dinotool.

        Parameters:
            dino_model (str): Name of the Dinov2 model to use for feature extraction.
            clip_model (str): Name of the CLIP model to use for feature extraction.
        """
        self.dino_model = dino_model
        self.clip_model = clip_model
        self.model = load_model(model_name=self.dino_model)
        self.extractor = DinoFeatureExtractor(self.model)
        self.transform_factory=TransformFactory(model_name=self.dino_model, patch_size=self.model.patch_size)
        self.pca = PCAModule(n_components=3)

    def extract_dino_features(self, image: Image.Image, filename: str = "single_file"):
        """
        Extracts features from a single image using the Dinov2 model.
        
        Parameters:
            image (Image.Image): The input image from which features are to be extracted.
            filename (str): The name of the file to be used for feature extraction. Default is "single_file".
        Returns:
            dict: A dictionary containing the filename, extracted features, input size, and feature map size.
        """
        if not image:
            raise Image.UnidentifiedImageError("No image provided for feature extraction.")
        
        #print(f"Processing image: {filename}")
        #print(f"Image size: {image.size}")


        transform = self.transform_factory.get_transform(image.size)
        input_size = transform.resize_size
        feature_map_size = transform.feature_map_size
        #print(f"Model input size: {input_size}")
        #print(f"Feature map size: {feature_map_size}")

        image_tensor = transform.transform(image).unsqueeze(0)
        features = self.extractor(image_tensor)
        return {
            "filename": filename,
            "features": features,
            "input_size": input_size,
            "feature_map_size": feature_map_size,
        } 
        

    def visualize_dino_features(self, image: Image.Image, features: LocalFeatures, input_size: tuple, feature_map_size: tuple, filename: str = "single_image", only_pca: bool = False, visualize: bool = False):
        """
        Visualizes the extracted features using PCA and displays the image.
        
        Parameters:
         image (Image.Image): The image from which features were extracted.
         features (LocalFeatures): The extracted features from the image. 
         input_size (tuple): The input size of the model used for feature extraction.
         feature_map_size (tuple): The size of the feature map.
         filename (str): The name of the file to be used for visualization. Default is "single_image".
         only_pca (bool): If True, only PCA visualization is shown. Default is False.
         visualize (bool): If True, displays the PCA visualization. Default is False.

        Returns:
         None: Displays the PCA visualization of the features.
        """
        if not image:
            raise Image.UnidentifiedImageError("No image provided for visualization.")
        
        self.pca.feature_map_size=feature_map_size
        self.pca.fit(features.flat().tensor, verbose=False)
        pca_array = self.pca.transform(features.flat().tensor, flattened=False)[0]
        
        frame = FrameData(img=image, features=features, pca=pca_array, filename=filename)

        out_img = frame_visualizer(
                    frame, output_size=input_size, only_pca=only_pca
                )
        
        if visualize:
            out_img.show()
        return out_img
    
    def visualize_dino_features_video(self, images: list[Image.Image], outfile_path: Path, framerate: int = 30, save_images=False):
        """
        Creates a video from the extracted feature images.
        
        Parameters:
            feature_images (list): List of (feature,image) to be included in the video.
            outfile_path (Path): Path where the video will be saved, ending in .mp4.
            framerate (int): Frame rate for the video. Default is 5.
            save_images (bool): If True, saves the feature images to a outfile_path.parent/"feature_images". Default is False.
            
        Returns:
            None: Saves the video to the specified outfile_path.
        """
        if not images:
            raise ValueError("No images provided for video creation.")
        
        if save_images:
            dir = outfile_path.parent.parent/ "feature_images" / f"{outfile_path.stem}"
            dir.mkdir(parents=True, exist_ok=True) if save_images else None
        else:
            tempdir = tempfile.TemporaryDirectory(dir=outfile_path.parent.parent / f"{outfile_path.stem}_images") 
            dir = Path(tempdir.name) 

        for idx, image in tqdm(enumerate(images), desc="Extracting features for video:", total=len(images)):
            filename=f"{idx:05d}"
            features = self.extract_dino_features(image=image, filename=filename)

            vis_img = self.visualize_dino_features(
                image=image,
                filename= features['filename'],
                features=features['features'],
                input_size=features['input_size'],
                feature_map_size=features['feature_map_size'],
                only_pca=False
            )
            image.close()
            filename = dir / f"{filename}.jpg"
            vis_img.save(filename)

        create_video_from_frames(
            str(dir),
            outfile_path,
            framerate= framerate
        )
       

if __name__ == "__main__":
    from utils.files import read_mcap_file
    
    import io
    
    # Example usage
    images = []
    
    f = "data/scantinel/250612_RG_dynamic_test_drive/IN002_MUL_SEN_0.2.0.post184+g6da4bed/20250612_144055642000_to_144155593000_CAM.mcap"
    f = Path(f)

    data = read_mcap_file(f, ["/camera"])
   
    for idx, msg in tqdm(enumerate(data), desc="Reading images", total=len(data), leave=False, unit="image"):
        image = Image.open(io.BytesIO(msg.proto_msg.data)).convert("RGB")
        images.append(image)

    extractor = Extractor()

    output_video_path = Path("data/scantinel/feature_vids")
    output_video_path.mkdir(parents=True, exist_ok=True)
    output_video_path = output_video_path / f"{f.stem}.mp4"

    extractor.visualize_dino_features_video(images=images, outfile_path=output_video_path, framerate=30, save_images=True)
