"""
Gradio Interface for Indonesian Herbal Plants Classification
"""
import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from models import get_model


class HerbalClassifier:
    """Inference class for herbal plant classification"""
    
    def __init__(self, model_name: str = None):
        self.device = config.DEVICE
        self.class_names = self._load_class_names()
        self.num_classes = len(self.class_names)
        
        # Load best model or specified model
        if model_name is None:
            model_name = self._find_best_model()
        
        self.model_name = model_name
        self.model = self._load_model(model_name)
        
        # Transform for inference
        self.transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _load_class_names(self):
        """Load class names from JSON"""
        class_names_path = config.OUTPUT_DIR / "class_names.json"
        if class_names_path.exists():
            with open(class_names_path, 'r') as f:
                return json.load(f)
        else:
            # Fallback to reading from data directory
            return sorted([d.name for d in config.DATA_DIR.iterdir() if d.is_dir()])
    
    def _find_best_model(self):
        """Find the best performing model"""
        results_path = config.OUTPUT_DIR / "training_results.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
            best_model = max(results.items(), key=lambda x: x[1].get('best_val_acc', 0))
            return best_model[0]
        return config.MODEL_NAMES[0]
    
    def _load_model(self, model_name: str):
        """Load a trained model"""
        model_path = config.MODELS_DIR / f"{model_name.lower()}.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model = get_model(model_name, self.num_classes, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print(f"Loaded model: {model_name}")
        print(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A')}")
        
        return model
    
    @torch.no_grad()
    def predict(self, image: Image.Image) -> tuple:
        """
        Predict class for a single image
        Returns: (predicted_class, confidence, all_probabilities)
        """
        # Preprocess
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        outputs = self.model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        
        # Get prediction
        confidence, predicted_idx = torch.max(probabilities, 0)
        predicted_class = self.class_names[predicted_idx.item()]
        
        # Get all probabilities as dict
        prob_dict = {
            self.class_names[i]: float(probabilities[i]) 
            for i in range(self.num_classes)
        }
        
        return predicted_class, float(confidence), prob_dict


# Global classifier instance
classifier = None


def load_classifier(model_name: str = None):
    """Load or reload classifier"""
    global classifier
    classifier = HerbalClassifier(model_name)
    return f"✅ Loaded model: {classifier.model_name}"


def classify_image(image, model_name: str):
    """Classify an uploaded image"""
    global classifier
    
    if image is None:
        return "Please upload an image", {}, ""
    
    # Load classifier if needed
    if classifier is None or (model_name and classifier.model_name != model_name):
        try:
            load_classifier(model_name if model_name else None)
        except Exception as e:
            return f"Error loading model: {str(e)}", {}, ""
    
    try:
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Predict
        predicted_class, confidence, prob_dict = classifier.predict(image)
        
        # Format result
        result_text = f"""
🌿 **Predicted Plant: {predicted_class.upper()}**
📊 **Confidence: {confidence * 100:.2f}%**

Model used: {classifier.model_name}
"""
        
        # Get top 5 predictions
        sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:5])
        
        # Herbal info
        herbal_info = get_herbal_info(predicted_class)
        
        return result_text, sorted_probs, herbal_info
        
    except Exception as e:
        return f"Error during prediction: {str(e)}", {}, ""


def get_herbal_info(plant_name: str) -> str:
    """Get information about the herbal plant"""
    herbal_database = {
        "jahe": "🌱 **Jahe (Zingiber officinale)**\n- Manfaat: Mengatasi mual, radang sendi, nyeri otot\n- Penggunaan: Minuman hangat, bumbu masakan, obat tradisional",
        "kunyit": "🌱 **Kunyit (Curcuma longa)**\n- Manfaat: Anti-inflamasi, antioksidan, kesehatan pencernaan\n- Penggunaan: Jamu kunyit asam, bumbu kari, obat maag",
        "kencur": "🌱 **Kencur (Kaempferia galanga)**\n- Manfaat: Mengatasi batuk, penambah nafsu makan\n- Penggunaan: Beras kencur, bumbu masakan",
        "lengkuas": "🌱 **Lengkuas (Alpinia galanga)**\n- Manfaat: Antibakteri, mengatasi masalah pencernaan\n- Penggunaan: Bumbu masakan, obat tradisional",
        "temulawak": "🌱 **Temulawak (Curcuma xanthorrhiza)**\n- Manfaat: Kesehatan hati, meningkatkan nafsu makan\n- Penggunaan: Jamu temulawak, suplemen kesehatan",
        "serai": "🌱 **Serai (Cymbopogon citratus)**\n- Manfaat: Relaksasi, mengurangi kembung, pengusir nyamuk\n- Penggunaan: Teh serai, bumbu masakan, aromaterapi",
        "daun salam": "🌱 **Daun Salam (Syzygium polyanthum)**\n- Manfaat: Menurunkan kolesterol, mengontrol gula darah\n- Penggunaan: Bumbu masakan, air rebusan",
        "cengkeh": "🌱 **Cengkeh (Syzygium aromaticum)**\n- Manfaat: Pereda nyeri gigi, antiseptik, meningkatkan imunitas\n- Penggunaan: Bumbu masakan, rokok kretek, obat sakit gigi",
        "kayu manis": "🌱 **Kayu Manis (Cinnamomum verum)**\n- Manfaat: Mengontrol gula darah, antioksidan\n- Penggunaan: Minuman hangat, bumbu kue, jamu",
        "pala": "🌱 **Pala (Myristica fragrans)**\n- Manfaat: Membantu tidur, mengurangi nyeri\n- Penggunaan: Bumbu masakan, minuman hangat",
        "lada": "🌱 **Lada (Piper nigrum)**\n- Manfaat: Meningkatkan pencernaan, antioksidan\n- Penggunaan: Bumbu masakan, pengobatan tradisional",
        "daun kemangi": "🌱 **Kemangi (Ocimum basilicum)**\n- Manfaat: Menyegarkan mulut, melancarkan pencernaan\n- Penggunaan: Lalapan, bumbu masakan",
        "bawang putih": "🌱 **Bawang Putih (Allium sativum)**\n- Manfaat: Antibakteri, menurunkan tekanan darah\n- Penggunaan: Bumbu masakan, obat tradisional",
        "bawang merah": "🌱 **Bawang Merah (Allium cepa)**\n- Manfaat: Menurunkan gula darah, antibakteri\n- Penggunaan: Bumbu masakan, obat tradisional",
        "kemiri": "🌱 **Kemiri (Aleurites moluccanus)**\n- Manfaat: Kesehatan rambut, sumber energi\n- Penggunaan: Bumbu masakan, minyak rambut",
    }
    
    plant_lower = plant_name.lower()
    if plant_lower in herbal_database:
        return herbal_database[plant_lower]
    
    return f"🌱 **{plant_name.title()}**\n- Informasi detail belum tersedia dalam database\n- Tanaman ini termasuk rempah-rempah Indonesia"


def get_available_models():
    """Get list of available trained models"""
    models = []
    for model_name in config.MODEL_NAMES:
        model_path = config.MODELS_DIR / f"{model_name.lower()}.pth"
        if model_path.exists():
            models.append(model_name)
    return models


def create_gradio_interface():
    """Create Gradio interface"""
    
    available_models = get_available_models()
    if not available_models:
        available_models = config.MODEL_NAMES
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-title {
        text-align: center;
        color: #2e7d32;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Indonesian Herbal Plants Classifier") as demo:
        gr.Markdown("""
        # 🌿 Indonesian Herbal Plants Classification
        ### Klasifikasi 31 Jenis Tanaman Herbal Indonesia menggunakan Deep Learning
        
        Upload gambar tanaman herbal dan sistem akan mengidentifikasi jenisnya beserta informasi khasiatnya.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="📷 Upload Gambar Tanaman",
                    type="pil",
                    height=300
                )
                
                model_dropdown = gr.Dropdown(
                    choices=available_models,
                    value=available_models[0] if available_models else None,
                    label="🤖 Pilih Model",
                    info="Pilih model deep learning untuk klasifikasi"
                )
                
                classify_btn = gr.Button("🔍 Identifikasi Tanaman", variant="primary", size="lg")
                
                gr.Markdown("""
                ### 📋 Daftar Tanaman yang Dapat Dikenali:
                adas, andaliman, asam jawa, bawang bombai, bawang merah, bawang putih, 
                biji ketumbar, bunga lawang, cengkeh, daun jeruk, daun kemangi, daun ketumbar, 
                daun salam, jahe, jinten, kapulaga, kayu manis, kayu secang, kemiri, kemukus, 
                kencur, kluwek, kunyit, lada, lengkuas, pala, saffron, serai, vanili, wijen
                """)
            
            with gr.Column(scale=1):
                result_text = gr.Markdown(label="📊 Hasil Prediksi")
                
                confidence_plot = gr.Label(
                    label="📈 Top 5 Probabilitas",
                    num_top_classes=5
                )
                
                herbal_info = gr.Markdown(label="🌱 Informasi Tanaman")
        
        # Example images
        gr.Markdown("### 📸 Contoh Gambar")
        gr.Examples(
            examples=[
                ["examples/jahe.jpg"],
                ["examples/kunyit.jpg"],
                ["examples/lengkuas.jpg"],
            ] if Path("examples").exists() else [],
            inputs=input_image,
            label="Klik untuk mencoba"
        )
        
        # Event handlers
        classify_btn.click(
            fn=classify_image,
            inputs=[input_image, model_dropdown],
            outputs=[result_text, confidence_plot, herbal_info]
        )
        
        input_image.change(
            fn=classify_image,
            inputs=[input_image, model_dropdown],
            outputs=[result_text, confidence_plot, herbal_info]
        )
        
        gr.Markdown("""
        ---
        ### 📖 Tentang Aplikasi
        - **5 Model Deep Learning**: YOLOv11, EfficientNetV2, ConvNeXtV2, ViT, Hybrid CNN-ViT
        - **31 Kelas Tanaman**: Rempah dan tanaman herbal Indonesia
        - **Dataset**: Indonesian Spices Dataset (6.510 gambar)
        
        Made with ❤️ for Indonesian Herbal Heritage
        """)
    
    return demo


# Create interface
demo = create_gradio_interface()


if __name__ == "__main__":
    # Launch the interface
    demo.launch(
        share=True,  # Create public link
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
