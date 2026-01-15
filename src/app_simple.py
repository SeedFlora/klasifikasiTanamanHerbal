"""
Simplified Gradio Interface for Indonesian Herbal Plants Classification
"""
import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from models import get_model


# Load class names
class_names_path = config.OUTPUT_DIR / "class_names.json"
with open(class_names_path, 'r') as f:
    class_names = json.load(f)

# Load model
device = config.DEVICE
model_path = config.MODELS_DIR / "efficientnetv2.pth"

print(f"Loading model from {model_path}")
checkpoint = torch.load(model_path, map_location=device)
model = get_model("efficientnetv2", len(class_names), pretrained=False)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
print(f"Model loaded! Best val acc: {checkpoint.get('best_val_acc', 'N/A')}")

# Transform
transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict(image):
    """Predict plant class from image"""
    if image is None:
        return None, "Please upload an image"

    try:
        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]

        # Get top 5
        top5_prob, top5_idx = torch.topk(probabilities, 5)

        # Format results
        results = {
            class_names[idx.item()]: float(prob)
            for prob, idx in zip(top5_prob, top5_idx)
        }

        # Get top prediction
        predicted_class = class_names[top5_idx[0].item()]
        confidence = float(top5_prob[0])

        info = f"""
## 🌿 Hasil Klasifikasi

**Tanaman Terdeteksi:** {predicted_class.upper()}
**Confidence:** {confidence * 100:.2f}%

### Informasi Tanaman
{get_herbal_info(predicted_class)}
"""

        return results, info

    except Exception as e:
        return None, f"Error: {str(e)}"


def get_herbal_info(plant_name: str) -> str:
    """Get information about the herbal plant"""
    herbal_database = {
        "jahe": "Jahe (Zingiber officinale) - Manfaat: Mengatasi mual, radang sendi, nyeri otot. Penggunaan: Minuman hangat, bumbu masakan.",
        "kunyit": "Kunyit (Curcuma longa) - Manfaat: Anti-inflamasi, antioksidan, kesehatan pencernaan. Penggunaan: Jamu, bumbu kari.",
        "kencur": "Kencur (Kaempferia galanga) - Manfaat: Mengatasi batuk, penambah nafsu makan. Penggunaan: Beras kencur, bumbu masakan.",
        "lengkuas": "Lengkuas (Alpinia galanga) - Manfaat: Antibakteri, mengatasi masalah pencernaan. Penggunaan: Bumbu masakan.",
        "serai": "Serai (Cymbopogon citratus) - Manfaat: Relaksasi, mengurangi kembung. Penggunaan: Teh serai, bumbu masakan.",
        "daun salam": "Daun Salam (Syzygium polyanthum) - Manfaat: Menurunkan kolesterol, mengontrol gula darah. Penggunaan: Bumbu masakan.",
        "cengkeh": "Cengkeh (Syzygium aromaticum) - Manfaat: Pereda nyeri gigi, antiseptik. Penggunaan: Bumbu masakan, obat sakit gigi.",
        "kayu manis": "Kayu Manis (Cinnamomum verum) - Manfaat: Mengontrol gula darah, antioksidan. Penggunaan: Minuman, bumbu kue.",
        "pala": "Pala (Myristica fragrans) - Manfaat: Membantu tidur, mengurangi nyeri. Penggunaan: Bumbu masakan, minuman hangat.",
        "lada": "Lada (Piper nigrum) - Manfaat: Meningkatkan pencernaan, antioksidan. Penggunaan: Bumbu masakan.",
        "daun kemangi": "Kemangi (Ocimum basilicum) - Manfaat: Menyegarkan mulut, melancarkan pencernaan. Penggunaan: Lalapan, bumbu.",
        "bawang putih": "Bawang Putih (Allium sativum) - Manfaat: Antibakteri, menurunkan tekanan darah. Penggunaan: Bumbu masakan.",
        "bawang merah": "Bawang Merah (Allium cepa) - Manfaat: Menurunkan gula darah, antibakteri. Penggunaan: Bumbu masakan.",
    }

    return herbal_database.get(plant_name.lower(), f"Tanaman herbal Indonesia: {plant_name}")


# Create interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="📷 Upload Gambar Tanaman Herbal", type="pil"),
    outputs=[
        gr.Label(label="📊 Top 5 Prediksi", num_top_classes=5),
        gr.Markdown(label="🌿 Informasi Tanaman")
    ],
    title="🌿 Indonesian Herbal Plants Classification",
    description="""
    ### Klasifikasi 31 Jenis Tanaman Herbal Indonesia menggunakan Deep Learning

    Upload gambar tanaman herbal dan sistem akan mengidentifikasi jenisnya beserta informasi khasiatnya.

    **Model:** EfficientNetV2-S (95.08% accuracy)

    **31 Tanaman yang dapat dikenali:**
    adas, andaliman, asam jawa, bawang bombai, bawang merah, bawang putih, biji ketumbar,
    bunga lawang, cengkeh, daun jeruk, daun kemangi, daun ketumbar, daun salam, jahe, jinten,
    kapulaga, kayu manis, kayu secang, kemiri, kemukus, kencur, kluwek, kunyit, lada, lengkuas,
    pala, saffron, serai, vanili, wijen
    """,
    article="""
    ---
    ### 📖 Tentang Aplikasi
    - **Model:** EfficientNetV2-S (95.08% accuracy)
    - **Dataset:** Indonesian Spices Dataset (6,510 gambar)
    - **Training:** 10 epochs, Mixed Precision (AMP), AdamW + OneCycleLR

    Made with ❤️ for Indonesian Herbal Heritage
    """,
    allow_flagging="never",
    theme=gr.themes.Soft(),
    live=False  # Disable auto-submit, require manual button click
)


if __name__ == "__main__":
    demo.launch()
