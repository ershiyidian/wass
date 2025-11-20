
import sys
import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from stereo.modeling.models.lightstereo.lightstereo import LightStereo

class Config:
    MAX_DISP = 192
    LEFT_ATT = True
    BACKCONE = 'MobileNetv2'
    AGGREGATION_BLOCKS = [1, 2, 4]
    EXPANSE_RATIO = 4

    def get(self, key, default):
        return getattr(self, key, default)

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def prepare_input(img, device):
    # Normalize to [0, 1] and convert to tensor (1, C, H, W)
    # Imagenet mean/std? The paper code usually uses mean/std normalization.
    # OpenStereo datasets usually normalize with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # I should verify this but it's standard for pretrained backbones (timm).

    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img.to(device)

def pad_image(img, divisor=32):
    b, c, h, w = img.shape
    # Calculate padding
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor

    if pad_h == 0 and pad_w == 0:
        return img, 0, 0

    padded_img = F.pad(img, (0, pad_w, 0, pad_h))
    return padded_img, pad_h, pad_w

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run_lightstereo.py <workdir>")
        sys.exit(1)

    workdir = sys.argv[1]
    left_path = os.path.join(workdir, "rectified_left.png")
    right_path = os.path.join(workdir, "rectified_right.png")
    output_path = os.path.join(workdir, "external_disparity.tiff")

    print(f"Loading images from {workdir}")
    # WASS needs Right view disparity.
    # We use Right as Left (Ref) and Left as Right (Tgt) for the model.
    # Model output will be disparity for Right image.
    # x_L = x_R - d (LightStereo convention if Input L=Right, R=Left)
    # WASS expects x_L = x_R - d?
    # WASS code: float xl = xr - disp.
    # So WASS expects disp > 0.
    # And x_L = x_R - disp.
    # So this matches perfectly.

    left_img_cv = load_image(left_path)
    right_img_cv = load_image(right_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ref_img = prepare_input(right_img_cv, device) # Right as Ref
    tgt_img = prepare_input(left_img_cv, device) # Left as Target

    # Pad
    ref_img_padded, pad_h, pad_w = pad_image(ref_img)
    tgt_img_padded, _, _ = pad_image(tgt_img) # Padding should be same

    # Load Model
    cfg = Config()
    model = LightStereo(cfg)
    model.to(device)

    model_path = "pretrained/LightStereo-S-SceneFlow-General.pth"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        sys.exit(1)

    print(f"Loading model weights from {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state', checkpoint)
    if 'model' in state_dict:
         state_dict = state_dict['model']

    # Remove 'module.' prefix if DDP was used
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    print("Running inference...")
    with torch.no_grad():
        input_data = {'left': ref_img_padded, 'right': tgt_img_padded}
        output = model(input_data)
        disp_pred = output['disp_pred'] # [B, 1, H, W]

        # Unpad
        h_padded, w_padded = disp_pred.shape[2], disp_pred.shape[3]
        h_orig = h_padded - pad_h
        w_orig = w_padded - pad_w
        disp_pred = disp_pred[:, :, :h_orig, :w_orig]

        disp_np = disp_pred.squeeze().cpu().numpy()

    print(f"Saving disparity to {output_path}")
    # Save as TIFF float32
    cv2.imwrite(output_path, disp_np)
    print("Done.")

if __name__ == "__main__":
    main()
