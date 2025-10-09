import torch
import torchvision
import pandas as pd
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import numpy as np
from waggle.plugin import Plugin
from waggle.data.vision import Camera


# ============================================================
# 1. Model and Metadata Setup
# ============================================================

WEIGHTS_PATH = "weights/model.pth"
CLASSES_PATH = "classes.csv"


def build_model():
    import argparse
    from torch.serialization import add_safe_globals
    add_safe_globals([argparse.Namespace])

    model = torchvision.models.regnet_y_32gf(weights=None)
    model.fc = torch.nn.Linear(3712, 2526)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load checkpoint safely
    ckpt = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)

    # Disable BatchNorm running stats
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.track_running_stats = False

    model.eval()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    return model, device


def load_classes():
    df = pd.read_csv(CLASSES_PATH)
    names = list(df["genus"] + " " + df["species"])
    roles = list(df["Role in Ecosystem"])
    return names, roles


# ============================================================
# 2. Image Preprocessing
# ============================================================

def preprocess_image(image_np):
    crop_size = 224
    resize_size = 256
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    interpolation = InterpolationMode.BILINEAR

    transforms_val = transforms.Compose([
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=mean, std=std),
    ])

    image = Image.fromarray(np.uint8(image_np))
    tensor = transforms_val(image).reshape((1, 3, 224, 224))  # keep reshape
    return tensor


# ============================================================
# 3. Inference + Publishing
# ============================================================

def main():
    model, device = build_model()
    scientific_names, roles = load_classes()

    with Plugin() as plugin, Camera() as camera:
        print("📸 InsectNet edge inference started — waiting for camera frames...")

        for snapshot in camera.stream():
            try:
                # Preprocess
                image_tensor = preprocess_image(snapshot.data)
                image_tensor = image_tensor.to(device, non_blocking=True)

                # Inference
                with torch.inference_mode():
                    output = model(image_tensor)
                    op = torch.nn.functional.softmax(output, dim=1)
                    op_ix = torch.argmax(op)
                    confidence = op[0][op_ix].item()
                    pred_index = op_ix.item()

                # Interpret result
                sci_name = scientific_names[pred_index]
                role = roles[pred_index]
                status = "OOD (low confidence)" if confidence < 0.97 else "ID"

                # Print for debugging
                print(f"🪲 {sci_name:40s} | {role:25s} | conf={confidence:.3f} | {status}")

                # Publish to Beehive
                plugin.publish("insectnet.prediction", sci_name, timestamp=snapshot.timestamp)
                plugin.publish("insectnet.confidence", confidence, timestamp=snapshot.timestamp)
                plugin.publish("insectnet.role", role, timestamp=snapshot.timestamp)
                plugin.publish("insectnet.status", status, timestamp=snapshot.timestamp)

                # Save and upload snapshot
                snapshot_filename = "snapshot.jpg"
                snapshot.save(snapshot_filename)
                plugin.upload_file(snapshot_filename, timestamp=snapshot.timestamp)

            except Exception as e:
                print(f"[⚠️ ERROR] Frame processing failed: {e}")


if __name__ == "__main__":
    main()
