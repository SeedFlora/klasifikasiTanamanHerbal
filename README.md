# Indonesian Herbal Plants Classification
## Klasifikasi Tanaman Herbal Indonesia Menggunakan Deep Learning dengan Arsitektur Hybrid CNN-Transformer

---

## 📚 Latar Belakang

Indonesia merupakan negara dengan kekayaan biodiversitas tanaman obat yang luar biasa, dengan lebih dari 30.000 spesies tanaman yang telah teridentifikasi, dimana sekitar 9.600 spesies diketahui memiliki khasiat obat [1]. Tanaman herbal telah menjadi bagian integral dari warisan budaya Indonesia, khususnya dalam pengobatan tradisional yang dikenal sebagai Jamu. Menurut WHO, sekitar 80% populasi di negara berkembang masih bergantung pada pengobatan tradisional berbasis tanaman untuk kebutuhan kesehatan primer mereka [2]. Namun, pengetahuan tentang identifikasi tanaman herbal semakin menurun di kalangan generasi muda, sementara kebutuhan akan sistem identifikasi otomatis yang akurat terus meningkat seiring dengan berkembangnya industri farmasi dan nutrasetikal berbasis tanaman herbal [3].

Perkembangan teknologi *deep learning* telah membawa revolusi signifikan dalam bidang klasifikasi citra, termasuk identifikasi tanaman. Penelitian oleh Mulugeta et al. (2024) dalam systematic review mereka yang dipublikasikan di *Frontiers in Plant Science* menunjukkan bahwa deep learning telah mencapai akurasi lebih dari 95% dalam klasifikasi tanaman obat, jauh melampaui metode machine learning tradisional [4]. Musyaffa et al. (2024) dalam penelitian "IndoHerb" berhasil mengembangkan sistem pengenalan tanaman obat Indonesia menggunakan transfer learning dengan akurasi mencapai 98.2% untuk 30 spesies tanaman herbal lokal [5]. Sementara itu, Bouakkaz et al. (2025) mendemonstrasikan bahwa optimasi arsitektur CNN dapat meningkatkan performa klasifikasi tanaman obat secara signifikan dengan penggunaan komputasi yang lebih efisien [6].

Arsitektur *Vision Transformer* (ViT) telah muncul sebagai alternatif yang menjanjikan untuk Convolutional Neural Network (CNN) dalam klasifikasi citra. Penelitian oleh Tonmoy et al. (2025) memperkenalkan "MobilePlantViT", sebuah arsitektur hybrid ViT yang dirancang khusus untuk klasifikasi penyakit tanaman pada perangkat mobile, mencapai keseimbangan optimal antara akurasi dan efisiensi komputasi [7]. Reedha et al. (2022) dalam studi mereka yang dipublikasikan di *Remote Sensing* menunjukkan bahwa Vision Transformer mengungguli CNN tradisional dalam klasifikasi tanaman dari citra UAV dengan margin 3-5% [8]. Lebih lanjut, penelitian oleh Sharma & Vardhan (2024) mengembangkan "AELGNet", sebuah jaringan berbasis attention yang menggabungkan fitur lokal dan global untuk klasifikasi daun tanaman obat dengan akurasi 97.8% [9].

Pendekatan *hybrid* yang mengkombinasikan CNN dengan Transformer telah menunjukkan hasil yang superior dalam berbagai tugas klasifikasi citra. Uma & Sarvika (2026) memperkenalkan "MediFlora-Net", sebuah model deep learning yang diperkuat dengan quantum computing untuk identifikasi tanaman obat presisi tinggi [10]. Penelitian oleh EM et al. (2025) yang dipublikasikan di *AgriEngineering* mengembangkan model hybrid untuk klasifikasi 39 spesies tanaman aromatik dan obat, mencapai akurasi 96.5% dengan dataset daun yang dikurasi secara khusus [11]. Sementara itu, Bhoyan et al. (2025) mengusulkan model dual-attention guided deep learning yang tidak hanya akurat tetapi juga interpretable, memberikan insight tentang bagaimana model membuat keputusan klasifikasi [12].

Efisiensi model deep learning menjadi pertimbangan penting dalam implementasi praktis. Mugisha et al. (2025) dalam penelitian mereka tentang edge computing untuk sistem IoT pertanian mendemonstrasikan bahwa knowledge distillation dari Vision Transformer ke model yang lebih ringan dapat mempertahankan 95% akurasi dengan pengurangan 70% parameter [13]. Islam et al. (2024) menggunakan Particle Swarm Optimization untuk mengoptimasi arsitektur cascaded network, mencapai akurasi 98.5% dalam klasifikasi tanaman obat dengan model yang 40% lebih efisien [14]. Penelitian oleh Azadnia et al. (2024) yang dipublikasikan di *Ecological Informatics* berhasil membedakan tanaman obat dari tanaman beracun menggunakan karakteristik visual daun dengan deep neural network, memberikan kontribusi penting untuk keamanan penggunaan tanaman [15].

Dataset dan benchmark yang berkualitas merupakan fondasi penting dalam penelitian klasifikasi tanaman. Jain et al. (2025) memperkenalkan "iNatAg", dataset skala besar dengan 4.7 juta gambar dari 2,959 spesies tanaman pertanian dan gulma, menjadi benchmark penting untuk evaluasi model [16]. Dalam konteks Indonesia, dataset Indonesian Spices dari Kaggle menyediakan 6,510 gambar dari 31 kelas rempah-rempah dan tanaman herbal Indonesia yang seimbang (balanced) [17]. Matin et al. (2025) dan Bijoy et al. (2025) masing-masing memperkenalkan dataset visual ekstensif dan "AI-MedLeafX" untuk diagnosis tanaman obat skala besar, memperkaya sumber daya untuk penelitian di bidang ini [18][19]. Barhate et al. (2024) dalam systematic review mereka di *Smart Agricultural Technology* mengidentifikasi gap penelitian dan arah masa depan untuk deteksi spesies tanaman menggunakan machine learning dan deep learning [20].

---

## 🎯 Rumusan Masalah

Berdasarkan latar belakang yang telah diuraikan, rumusan masalah dalam penelitian ini adalah:

1. Bagaimana mengembangkan sistem klasifikasi otomatis untuk 31 jenis tanaman herbal Indonesia menggunakan pendekatan deep learning?
2. Bagaimana perbandingan performa antara arsitektur CNN murni (EfficientNetV2, ConvNeXtV2), Transformer murni (Vision Transformer), dan arsitektur hybrid (CNN-ViT) dalam klasifikasi tanaman herbal?
3. Bagaimana efektivitas teknik transfer learning dan data augmentation dalam meningkatkan akurasi klasifikasi pada dataset tanaman herbal Indonesia?
4. Bagaimana mengimplementasikan model terbaik ke dalam aplikasi yang dapat diakses publik melalui platform Hugging Face dan Gradio?

---

## 🎯 Tujuan Penelitian

### Tujuan Umum
Mengembangkan sistem klasifikasi citra tanaman herbal Indonesia berbasis deep learning dengan arsitektur state-of-the-art tahun 2025 yang akurat dan dapat diimplementasikan dalam aplikasi praktis.

### Tujuan Khusus
1. Mengimplementasikan dan mengevaluasi 5 arsitektur deep learning terbaru: YOLOv11-cls, EfficientNetV2-S, ConvNeXtV2-Tiny, Vision Transformer (ViT), dan Hybrid CNN-ViT.
2. Membandingkan performa model menggunakan metrik evaluasi komprehensif: Accuracy, Precision, Recall, F1-Score, dan AUC-ROC.
3. Menganalisis confusion matrix dan ROC curve untuk memahami karakteristik klasifikasi setiap model.
4. Mengembangkan aplikasi web interaktif menggunakan Gradio untuk demonstrasi sistem klasifikasi.
5. Mempublikasikan model terlatih ke Hugging Face Hub untuk aksesibilitas dan reprodusibilitas penelitian.

---

## 💡 Manfaat Penelitian

### Manfaat Teoritis
1. Memberikan kontribusi pada pengembangan ilmu pengetahuan di bidang computer vision dan deep learning untuk klasifikasi tanaman herbal.
2. Menyediakan benchmark perbandingan arsitektur CNN, Transformer, dan Hybrid untuk domain klasifikasi tanaman Indonesia.
3. Memperkaya literatur penelitian tentang penerapan transfer learning pada dataset tanaman herbal lokal.

### Manfaat Praktis
1. **Bagi Masyarakat**: Menyediakan alat bantu identifikasi tanaman herbal yang mudah digunakan untuk edukasi dan pelestarian pengetahuan tradisional.
2. **Bagi Industri Farmasi**: Membantu proses quality control dan identifikasi bahan baku tanaman obat.
3. **Bagi Petani dan Pelaku UMKM**: Mempermudah identifikasi dan klasifikasi tanaman herbal untuk keperluan budidaya dan perdagangan.
4. **Bagi Peneliti**: Menyediakan model pre-trained dan kode sumber yang dapat digunakan untuk penelitian lanjutan.

---

## ⚠️ Batasan Penelitian

1. **Batasan Dataset**: Penelitian ini menggunakan Indonesian Spices Dataset dengan 31 kelas tanaman herbal dan 6,510 gambar. Dataset tidak mencakup seluruh spesies tanaman herbal Indonesia.
2. **Batasan Citra**: Model dilatih pada gambar 2D dengan resolusi 224×224 piksel. Variasi kondisi pencahayaan, sudut pengambilan, dan background pada data real-world mungkin mempengaruhi performa.
3. **Batasan Arsitektur**: Penelitian fokus pada 5 arsitektur terpilih dan tidak mencakup semua arsitektur deep learning yang tersedia.
4. **Batasan Hardware**: Training dilakukan dengan resource komputasi terbatas yang mempengaruhi hyperparameter tuning dan epoch training.
5. **Batasan Deployment**: Aplikasi Gradio memerlukan koneksi internet dan tidak dioptimasi untuk penggunaan offline di perangkat mobile.

---

## 🌿 Project Overview

Proyek klasifikasi 31 jenis tanaman herbal Indonesia menggunakan 5 model deep learning terbaru (2025).

## 📊 Models

| No | Model | Architecture | Description |
|----|-------|--------------|-------------|
| 1 | YOLOv11-cls | CNN-based | Fast and efficient classification |
| 2 | EfficientNetV2-S | CNN | Optimized CNN architecture |
| 3 | ConvNeXtV2-Tiny | Pure CNN | Modern CNN with transformer-style design |
| 4 | ViT-Base-16 | Transformer | Pure attention-based model |
| 5 | Hybrid-CNN-ViT | CNN + Transformer | CoAtNet-style hybrid architecture |

## 📁 Project Structure

```
skripsi tanaman herbal/
├── dataset/
│   └── Indonesian Spices Dataset/
│       ├── adas/
│       ├── andaliman/
│       ├── ... (31 classes)
│       └── wijen/
├── src/
│   ├── config.py          # Configuration
│   ├── dataset.py         # Data loading & augmentation
│   ├── models.py          # 5 model architectures
│   ├── trainer.py         # Training pipeline
│   ├── evaluator.py       # Evaluation & visualization
│   ├── app.py             # Gradio interface
│   ├── huggingface_upload.py  # HuggingFace upload
│   └── main.py            # Main pipeline
├── outputs/
│   ├── models/            # Trained models (.pth)
│   ├── plots/             # Visualizations
│   └── logs/              # Training logs
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision timm ultralytics transformers gradio huggingface_hub scikit-learn matplotlib seaborn pandas numpy pillow tqdm tensorboard
```

### 2. Run Training & Evaluation

```bash
cd src
python main.py
```

### 3. Run Gradio Interface

```bash
python app.py
```

### 4. Push to Hugging Face

```bash
python huggingface_upload.py --username YOUR_USERNAME --token YOUR_HF_TOKEN
```

## 📈 Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: Macro & weighted precision
- **Recall**: Macro & weighted recall
- **F1-Score**: Macro & weighted F1
- **AUC-ROC**: Area under ROC curve

## 🌱 Classes (31 Indonesian Herbal Plants)

1. adas
2. andaliman
3. asam jawa
4. bawang bombai
5. bawang merah
6. bawang putih
7. biji ketumbar
8. bukan rempah
9. bunga lawang
10. cengkeh
11. daun jeruk
12. daun kemangi
13. daun ketumbar
14. daun salam
15. jahe
16. jinten
17. kapulaga
18. kayu manis
19. kayu secang
20. kemiri
21. kemukus
22. kencur
23. kluwek
24. kunyit
25. lada
26. lengkuas
27. pala
28. saffron
29. serai
30. vanili
31. wijen

## 📊 Dataset

- **Source**: [Indonesian Spices Dataset (Kaggle)](https://www.kaggle.com/datasets/albertnathaniel12/indonesian-spices-dataset)
- **Total Images**: 6,510
- **Images per Class**: 210 (perfectly balanced)
- **Train/Val/Test Split**: 70%/15%/15%

## 🔧 Configuration

Edit `src/config.py` to modify:
- `IMAGE_SIZE`: Input image size (default: 224)
- `BATCH_SIZE`: Batch size (default: 32)
- `EPOCHS`: Number of epochs (default: 20)
- `LEARNING_RATE`: Learning rate (default: 1e-4)

## 📝 License

Apache 2.0

## 👤 Author

Seed Flora

---

## 📖 Referensi

[1] Elfahmi, H. J. Woerdenbag, and O. Kayser, "Jamu: Indonesian traditional herbal medicine towards rational phytopharmacological use," *Journal of Herbal Medicine*, vol. 4, no. 2, pp. 51-73, 2014. DOI: 10.1016/j.hermed.2014.01.002

[2] World Health Organization, "WHO Traditional Medicine Strategy 2014-2023," WHO Press, Geneva, 2013.

[3] M. S. I. Musyaffa, N. Yudistira, M. A. Rahman, and J. Batoro, "IndoHerb: Indonesia medicinal plants recognition using transfer learning and deep learning," *Heliyon*, vol. 10, no. 23, e40006, Dec. 2024. DOI: 10.1016/j.heliyon.2024.e40006

[4] A. K. Mulugeta, D. P. Sharma, and A. H. Mesfin, "Deep learning for medicinal plant species classification and recognition: a systematic review," *Frontiers in Plant Science*, vol. 14, 1286088, Jan. 2024. DOI: 10.3389/fpls.2023.1286088

[5] M. S. I. Musyaffa, N. Yudistira, M. A. Rahman, and J. Batoro, "IndoHerb: Indonesia Medicinal Plants Recognition using Transfer Learning and Deep Learning," *arXiv preprint arXiv:2308.01604*, 2023.

[6] H. Bouakkaz, M. Bouakkaz, C. A. Kerrache, and S. Dhelim, "Enhanced classification of medicinal plants using deep learning and optimized CNN architectures," *Heliyon*, vol. 11, no. 4, e25765, Feb. 2025. DOI: 10.1016/j.heliyon.2025.e25765

[7] M. R. Tonmoy, M. M. Hossain, N. Dey, and M. F. Mridha, "MobilePlantViT: A Mobile-friendly Hybrid ViT for Generalized Plant Disease Image Classification," *arXiv preprint arXiv:2503.16628*, Mar. 2025.

[8] R. Reedha, E. Dericquebourg, R. Canals, and A. Hafiane, "Vision Transformers for Weeds and Crops Classification of High Resolution UAV Images," *Remote Sensing*, vol. 14, no. 3, p. 592, 2022. DOI: 10.3390/rs14030592

[9] S. Sharma and M. Vardhan, "AELGNet: Attention-based Enhanced Local and Global Features Network for medicinal leaf and plant classification," *Computers in Biology and Medicine*, vol. 184, p. 109424, Jan. 2025. DOI: 10.1016/j.compbiomed.2024.109424

[10] K. V. Uma and P. Sarvika, "MediFlora-Net: Quantum-enhanced deep learning for precision medicinal plant identification," *Computational Biology and Chemistry*, vol. 116, p. 108354, Feb. 2026. DOI: 10.1016/j.compbiolchem.2025.108354

[11] S. EM, D. A. Chandy, S. PM, and A. Poulose, "A Hybrid Deep Learning Model for Aromatic and Medicinal Plant Species Classification Using a Curated Leaf Image Dataset," *AgriEngineering*, vol. 7, no. 8, p. 243, 2025. DOI: 10.3390/agriengineering7080243

[12] F. H. Bhoyan, M. H. K. Mehedi, and M. F. Mridha, "An efficient dual-attention guided deep learning model with interpretability for identifying medicinal plants," *Current Plant Biology*, vol. 42, p. 100401, Dec. 2025. DOI: 10.1016/j.cpb.2025.100401

[13] S. Mugisha, R. Kisitu, and F. Tushabe, "Hybrid Knowledge Transfer through Attention and Logit Distillation for On-Device Vision Systems in Agricultural IoT," *arXiv preprint arXiv:2504.16128*, Apr. 2025.

[14] M. T. Islam, W. Rahman, M. S. Hossain, and K. Roksana, "Medicinal Plant Classification Using Particle Swarm Optimized Cascaded Network," *IEEE Access*, vol. 12, pp. 45678-45692, 2024. DOI: 10.1109/ACCESS.2024.3374026

[15] R. Azadnia, F. Noei-Khodabadi, and M. Omid, "Medicinal and poisonous plants classification from visual characteristics of leaves using computer vision and deep neural networks," *Ecological Informatics*, vol. 82, p. 102655, Sep. 2024. DOI: 10.1016/j.ecoinf.2024.102655

[16] N. Jain, A. Joshi, and M. Earles, "iNatAg: Multi-Class Classification Models Enabled by a Large-Scale Benchmark Dataset with 4.7M Images of 2,959 Crop and Weed Species," *arXiv preprint arXiv:2503.20068*, Mar. 2025.

[17] A. Nathaniel, "Indonesian Spices Dataset," Kaggle, 2023. [Online]. Available: https://www.kaggle.com/datasets/albertnathaniel12/indonesian-spices-dataset

[18] M. M. H. Matin, M. Sefatullah, and H. B. Habib, "An extensive visual data for reliable identification of medicinal plant leaves," *Data in Brief*, vol. 59, p. 111285, Dec. 2025. DOI: 10.1016/j.dib.2025.111285

[19] M. F. Ferdous, F. B. K. Nissan, and M. H. I. Bijoy, "AI-MedLeafX: a large-scale computer vision dataset for medicinal plant diagnosis," *Data in Brief*, vol. 58, p. 111199, Oct. 2025. DOI: 10.1016/j.dib.2025.111199

[20] D. Barhate, S. Pathak, and A. K. Dubey, "A systematic review of machine learning and deep learning approaches in plant species detection," *Smart Agricultural Technology*, vol. 9, p. 100607, Dec. 2024. DOI: 10.1016/j.atech.2024.100607
