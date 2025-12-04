from typing import Dict
import os

import requests
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from cog import BasePredictor, Path, Input
from PIL import Image

# те же классы, что в inference.py (если в твоём inference.py другие – поправь тут)
class_names = ["Heart", "Oblong", "Oval", "Round", "Square"]

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "model_85_nn_.pth"
# тот же URL, что в оригинальном inference.py на Hugging Face
MODEL_URL = (
    "https://huggingface.co/fahd9999/model_85_nn_/resolve/main/model_85_nn_.pth?download=true"
)


def download_model_if_not_exists(model_url: str, model_path: str) -> None:
    """Качаем веса, если их ещё нет рядом с predict.py."""
    if os.path.exists(model_path):
        return

    print(f"Downloading model from {model_url}...")
    resp = requests.get(model_url, stream=True)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(model_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if not chunk:
                continue
            f.write(chunk)
            downloaded += len(chunk)
            # простой прогресс в лог
            if total:
                done = int(50 * downloaded / total)
                print("\r[{}{}]".format("=" * done, " " * (50 - done)), end="")
    print("\nModel downloaded.")


def preprocess_image(image_path: str) -> torch.Tensor:
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def get_probabilities(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    return probs * 100.0  # в проценты


class Predictor(BasePredictor):
    def setup(self):
        """Вызывается один раз при старте контейнера — скачиваем и загружаем модель."""
        download_model_if_not_exists(MODEL_URL, MODEL_PATH)

        self.model = torch.load(MODEL_PATH, map_location="cpu")
        self.model.eval()
        self.model.to(device)

    def predict(
        self,
        image: Path = Input(description="Фото лица для определения формы"),
    ) -> Dict[str, float]:
        """Инференс по одному изображению."""

        img_tensor = preprocess_image(str(image)).to(device)

        with torch.inference_mode():
            logits = self.model(img_tensor)

        percentages = get_probabilities(logits)[0]

        # вернём словарь "класс -> процент"
        return {
            class_names[i]: float(percentages[i].item())
            for i in range(len(class_names))
        }
