# 📋 Hugging Face Space Upload Checklist

## 🔗 URL Space Anda
https://huggingface.co/spaces/SeedFlora/indonesian-herbal-classifier

---

## ✅ Step 1: Siapkan Files

Buka Windows Explorer ke folder project:
```
d:\skripsi tanaman herbal\
```

---

## ✅ Step 2: Upload Root Files (3 files)

Ke HF Space → Tab "Files" → "Add file" → "Upload files"

Upload files ini:

### File 1: app.py
- [x] **app.py** (main entry point)
  - Path: `d:\skripsi tanaman herbal\app.py`
  - Size: ~500 bytes

### File 2: requirements.txt
- [x] **requirements.txt** (dependencies)
  - Path: `d:\skripsi tanaman herbal\requirements.txt`
  - Size: ~200 bytes

### File 3: README.md
- [x] **README.md**
  - ⚠️ RENAME dari `README_HF.md` → `README.md`
  - Path: `d:\skripsi tanaman herbal\README_HF.md`
  - Size: ~5 KB

---

## ✅ Step 3: Upload Folder src/ (8 files)

**Cara upload folder:**
1. Klik "Add file" → "Upload files"
2. Drag folder `src/` atau pilih semua file di dalamnya

Upload semua files di `d:\skripsi tanaman herbal\src\`:

- [x] **src/app.py** (Gradio interface) - 10 KB
- [x] **src/config.py** (Configuration) - 2 KB
- [x] **src/models.py** (Model definitions) - 15 KB
- [x] **src/dataset.py** (Data loader) - 5 KB
- [x] **src/trainer.py** (Training pipeline) - 12 KB
- [x] **src/evaluator.py** (Evaluation) - 8 KB
- [x] **src/huggingface_upload.py** (HF upload util) - 3 KB
- [x] **src/main.py** (Main pipeline) - 3 KB

---

## ✅ Step 4: Upload Folder outputs/ (3-5 files)

### Outputs Results (REQUIRED):

- [x] **outputs/training_results.json**
  - Path: `d:\skripsi tanaman herbal\outputs\training_results.json`
  - Size: 1 KB
  - ✅ Contains all model performance metrics

- [x] **outputs/class_names.json** (jika ada)
  - Path: `d:\skripsi tanaman herbal\outputs\class_names.json`
  - Size: 500 bytes
  - ℹ️ Class names list (31 plants)

---

### Outputs Models (PILIH MINIMAL 1):

**Rekomendasi: Upload 1-2 model terbaik untuk save space**

#### Model Terbaik (UPLOAD INI):
- [x] **outputs/models/efficientnetv2.pth** ⭐ RECOMMENDED
  - Path: `d:\skripsi tanaman herbal\outputs\models\efficientnetv2.pth`
  - Size: 78 MB
  - Accuracy: 95.08% (BEST)

#### Optional (pilih salah satu atau semua):
- [ ] **outputs/models/yolov11.pth**
  - Size: 78 MB
  - Accuracy: 95.08% (BEST - sama dengan efficientnetv2)

- [ ] **outputs/models/convformer.pth**
  - Size: 101 MB
  - Accuracy: 94.77%

- [ ] **outputs/models/convnextv2.pth**
  - Size: 107 MB
  - Accuracy: 93.95%

- [ ] **outputs/models/internimage.pth**
  - Size: 108 MB
  - Accuracy: 89.86%

**💡 TIP**: Untuk start, cukup upload **efficientnetv2.pth** saja (model terbaik & paling ringan)

---

## ✅ Step 5: Verifikasi File Structure

Setelah upload, struktur di HF Space harus seperti ini:

```
indonesian-herbal-classifier/
├── README.md
├── app.py
├── requirements.txt
├── src/
│   ├── app.py
│   ├── config.py
│   ├── dataset.py
│   ├── evaluator.py
│   ├── huggingface_upload.py
│   ├── main.py
│   ├── models.py
│   └── trainer.py
└── outputs/
    ├── training_results.json
    ├── class_names.json (optional)
    └── models/
        └── efficientnetv2.pth (minimal 1 model)
```

---

## ✅ Step 6: Wait for Build

1. Setelah upload selesai, HF akan auto-build
2. Lihat di tab "Logs" untuk monitor progress
3. Status berubah: "Building" → "Running"
4. Biasanya butuh 2-5 menit

---

## ✅ Step 7: Test App

1. Klik tab "App"
2. Upload gambar tanaman herbal (dari `dataset/` folder)
3. Pilih model: efficientnetv2
4. Klik "Identifikasi Tanaman"
5. Cek hasil prediksi

---

## 🔧 Troubleshooting

### Error: "No module named 'src'"
- ✅ Fix: Pastikan folder `src/` diupload dengan benar
- ✅ Check: File structure harus persis seperti di Step 5

### Error: "Model not found"
- ✅ Fix: Pastikan minimal 1 file .pth di `outputs/models/`
- ✅ Recommended: Upload `efficientnetv2.pth`

### Error: "Invalid YAML metadata"
- ✅ Fix: README.md harus punya header YAML di atas
- ✅ Copy dari `README_HF.md` yang sudah fixed

### App not loading
- ✅ Check Logs tab untuk error messages
- ✅ Restart Space: Settings → "Factory reboot"

---

## 📦 Quick Upload Summary

**Minimal Files (untuk quick test):**
```
✅ app.py
✅ requirements.txt
✅ README.md (from README_HF.md)
✅ src/ (all 8 files)
✅ outputs/training_results.json
✅ outputs/models/efficientnetv2.pth
```

**Total size minimal**: ~80 MB

**Full Upload (all models):**
Total size: ~550 MB (termasuk 5 models)

---

## 🎯 Final Check

Sebelum declare success, test:

1. ✅ App loading tanpa error
2. ✅ Upload gambar works
3. ✅ Model prediction works
4. ✅ Hasil prediksi benar (confidence > 80%)
5. ✅ Info tanaman muncul

---

## 🌐 Links

- **Your Space**: https://huggingface.co/spaces/SeedFlora/indonesian-herbal-classifier
- **Logs**: https://huggingface.co/spaces/SeedFlora/indonesian-herbal-classifier/logs
- **Settings**: https://huggingface.co/spaces/SeedFlora/indonesian-herbal-classifier/settings

---

## 📞 Need Help?

Jika ada masalah:
1. Check Logs tab
2. Read error message
3. Google error message
4. Ask di HF Community: https://discuss.huggingface.co/

---

Good luck! 🚀🌿
