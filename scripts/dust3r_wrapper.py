import torch
from models.dust3r.inference import inference
from models.dust3r.model import AsymmetricCroCo3DStereo
from models.dust3r.utils.image import load_images
from models.dust3r.image_pairs import make_pairs
from utils.misc import setup_logger

logger = setup_logger("dust3r")

class Dust3RWrapper:
    def __init__(self, model_name="naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(self.device).eval()

    def predict_pointmap(self, pil_img):
        img_list = load_images([pil_img, pil_img], 512)
        pairs = make_pairs(img_list, scene_graph='complete', prefilter=None, symmetrize=True)
        logger.info(f"Loaded {len(pairs)} image pairs for inference.")
        output = inference(pairs, self.model, device=self.device, batch_size=1)
        out= output['pred1']['pts3d'].squeeze(0).cpu().numpy()
        #print(f"[Dust3RWrapper] >> Predicted point cloud shape: {out.shape}")
        return out.reshape(-1, 3)
  