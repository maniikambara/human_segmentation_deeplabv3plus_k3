# Segmentasi Manusia dengan DeepLabV3+

Segmentasi biner manusia/latar belakang menggunakan arsitektur DeepLabV3+ dengan
encoder ResNet50 pretrained (ImageNet). Setiap piksel diklasifikasikan sebagai
manusia (1) atau latar belakang (0) pada resolusi 512x512. Project ini dioptimalkan
untuk pelatihan lokal pada satu GPU kelas menengah (dirancang dan diuji pada
RTX 4050 dengan 6 GB VRAM dan 16 GB RAM sistem).

Project ini disusun sebagai **notebook modular, satu per komponen**. Setiap
notebook mendokumentasikan dan menjalankan bagiannya dari pipeline dengan
visualisasi matplotlib di setiap tahap. Notebook dependen mengambil apa yang
dibutuhkan dari notebook fondasi secara otomatis, sehingga hanya ada satu
sumber kebenaran.

---

## Notebook

| Notebook | Peran | Bergantung pada |
|---|---|---|
| `metrics.ipynb` | Loss dan metrik (Dice, Focal, kombinasi, IoU) + Boundary IoU | tidak ada |
| `model.ipynb` | Arsitektur DeepLabV3+ + helper pembekuan backbone | tidak ada |
| `data.ipynb` | Augmentasi, pipeline `tf.data`, dan plot EDA | tidak ada |
| `segmentation_prep.ipynb` | Langkah 1: standardisasi dataset (rename semua gambar) | tidak ada |
| `train.ipynb` | Langkah 3: pelatihan | metrics, model, data |
| `eval.ipynb` | Langkah 4: evaluasi | data, metrics |
| `predict.ipynb` | Langkah 5: inferensi pada gambar baru | data |

### Bagaimana notebook saling terhubung

Notebook fondasi (`metrics`, `model`, `data`) hanya mendefinisikan fungsi.
Notebook dependen memuat definisi tersebut di bagian atas dengan magic `%run`
dari Jupyter, contohnya di `train.ipynb`:

```python
_DEMO = False
%run metrics.ipynb
%run model.ipynb
%run data.ipynb
_DEMO = True
```

Flag `_DEMO` adalah mekanisme kecil yang membuat ini bersih. Setiap notebook
fondasi diakhiri dengan sel demo atau self-check yang dijaga oleh `if _DEMO:`.
Ketika Anda membuka notebook tersebut secara mandiri, `_DEMO` default ke `True`
dan demo berjalan. Ketika notebook lain memuatnya dengan `%run`, pemanggil
menyetel `_DEMO = False` terlebih dahulu, sehingga hanya definisi yang dimuat
dan demo berat dilewati.

Ini berarti Anda bisa menjalankan notebook fondasi secara mandiri untuk
memeriksanya, atau cukup menjalankan notebook dependen dari atas ke bawah dan
membiarkannya mengambil dependensinya sendiri.

---

## Urutan eksekusi

1. **`segmentation_prep.ipynb`** - mengganti nama semua gambar dalam dataset
   (5.110 train + 568 val). Langkah ini dijalankan sekali; dijaga oleh `RUN_PREP = False`.
2. **`data.ipynb`** - jalankan secara mandiri untuk membangun dataset augmentasi
   dan plot EDA (demo-nya menulis ke `new_data/`).
3. **`train.ipynb`** - melatih dan menyimpan checkpoint terbaik ke `files/model.h5`.
   Notebook ini memuat `metrics`, `model` dan `data` via `%run`, jadi Anda tidak
   perlu menjalankannya terlebih dahulu.
4. **`eval.ipynb`** - mengevaluasi checkpoint pada set tes yang disisihkan.
5. **`predict.ipynb`** - melakukan segmentasi pada gambar yang Anda tempatkan di
   `test_images/image/`.
6. **`app.py`** - halaman web Streamlit untuk mencoba model secara interaktif
   (lihat "Aplikasi uji coba" di bawah).

---

## Aplikasi uji coba (Streamlit)

`app.py` adalah halaman web ringan untuk mencoba checkpoint yang sudah dilatih:
unggah satu foto, hasil segmentasi langsung tampil tanpa tombol submit.

```bash
streamlit run app.py
```

Butuh `files/model.h5` (hasil `train.ipynb`) sudah ada; jika belum, halaman
menampilkan pesan untuk menjalankan training terlebih dahulu. Fitur:

- Tiga mode tampilan - **Overlay** (warna di atas orang, dengan slider kekuatan
  warna), **Potongan** (latar belakang dihapus), **Mask** (mask biner hitam/putih).
- Tombol unduh untuk menyimpan hasil sebagai PNG.
- Preprocessing identik dengan `predict.ipynb` (lihat `inference.py`, modul plain
  Python yang dipakai bersama agar aplikasi tidak perlu memuat notebook).

Jika Anda mau, buka satu sesi Jupyter dan jalankan notebook sesuai urutan ini;
dependensi `%run` membuat setiap notebook dependen mandiri tanpa memandang
urutannya.

---

## Struktur project

```
.
├── metrics.ipynb              # loss + metrik (fondasi)
├── model.ipynb               # arsitektur (fondasi)
├── data.ipynb                # augmentasi + pipeline + EDA (fondasi)
├── segmentation_prep.ipynb   # Langkah 1 - standardisasi dataset
├── train.ipynb               # Langkah 3 - pelatihan
├── eval.ipynb                # Langkah 4 - evaluasi
├── predict.ipynb             # Langkah 5 - inferensi
│
├── people_segmentation/      # Dataset mentah (diunduh dari Kaggle)
│   ├── images/
│   ├── masks/
│   └── segmentation/
│
├── new_data/                 # Dataset augmentasi (dibuat oleh data.ipynb)
│   ├── train/image|mask/
│   └── test/image|mask/
│
├── files/                    # Artefak pelatihan (model.h5, data.csv, score.csv)
├── results/                  # Strip perbandingan evaluasi
├── test_images/              # I/O inferensi (image/ masuk, mask/ keluar)
└── plots/                    # Semua figur matplotlib (dibuat otomatis)
```

---

## Arsitektur

| Komponen | Detail |
|---|---|
| Encoder | ResNet50 pretrained pada ImageNet (kedalaman efektif: hingga `conv4_block6_out`) |
| Skip connection | `conv1_relu` (256x256x64), `conv2_block3_out` (128x128x256) |
| Cabang ASPP | 6 - global pool, konv 1x1, konv separable terdilasi (rate 6/12/18/24) + Dropout(0.3) |
| Tahap decoder | 3 - 32x32 -> 128x128 -> 256x256 -> 512x512 |
| Atensi | Squeeze-and-Excitation setelah konkatenasi decoder |
| Head output | konv 1x1 + sigmoid, keduanya dipaku ke float32; threshold 0.5 |
| Loss | Combined Loss = Dice Loss + Focal Loss (gamma=2, alpha=0.25) |

**Jumlah parameter.** Model yang dirakit memiliki sekitar 10,9 juta parameter,
bukan 23,5 juta dari ResNet50 penuh. Tap terdalam yang digunakan oleh DeepLab
adalah `conv4_block6_out`, sehingga Keras memangkas tahap `conv5` yang tidak
digunakan secara otomatis. Dari jumlah tersebut, sekitar 2,27 juta berada di
ASPP dan head decoder; sisanya di backbone.

**Head output float32.** Di bawah kebijakan mixed_float16, konvolusi 1x1 terakhir
dan sigmoid dipaku ke `dtype="float32"`. Ini menjaga peta probabilitas dalam
presisi penuh dan bekerja pada Keras 2 (TF 2.10-2.12) maupun Keras 3.

---

## Dataset

**People Segmentation Dataset** - [Nikhil Tomar](https://www.kaggle.com/datasets/nikhilroxtomar/person-segmentation),
5.678 pasangan gambar-mask.

`segmentation_prep.ipynb` menggunakan **semua** gambar tanpa seleksi, hanya
mengganti nama menjadi `photo-1 ... photo-N` dan memperbarui file pembagian.

| | Jumlah |
|---|---|
| Total gambar | 5.678 |
| Train | 5.110 |
| Validation | 568 |
| Pasangan pelatihan augmentasi | ~40.880 (8 varian masing-masing) |

> Catatan: `segmentation_prep.ipynb` mengganti nama file dan menulis ulang
> dataset di tempat. Simpan salinan terpisah dari unduhan asli jika Anda ingin
> memulai ulang nantinya.

**Encoding mask.** Mask mentah dataset ini menyimpan foreground sebagai piksel
bernilai `1` (bukan `255`) — array 2D dengan nilai unik `{0, 1}`. Semua
binarisasi mask di pipeline (`augment_data`, `read_mask`, evaluasi) memakai
ambang `> 0`, bukan `> 127`, justru karena itu: mengasumsikan `{0, 255}` akan
membuat setiap mask jadi kosong secara diam-diam (model lalu "sukses" belajar
menebak seluruh gambar sebagai latar belakang, ditandai oleh `iou≈1.0` namun
`recall`/`precision` tetap 0 sepanjang training). Jika mengganti dataset,
periksa dulu rentang nilai mask sebelum mengandalkan ambang apa pun.

---

## Preprocessing (konsisten train / eval / inferensi)

Setiap gambar dan mask diubah ukurannya ke 512x512 dengan resize biasa:
`INTER_AREA` untuk gambar dan `INTER_NEAREST` untuk mask. Fungsi
`standardize_image` / `standardize_mask` yang sama (didefinisikan di `data.ipynb`)
digunakan pada saat pelatihan, evaluasi, dan inferensi.

Ini menghilangkan ketidakcocokan train/serve di pipeline asli, di mana pelatihan
melakukan center-crop (membuang pinggiran gambar) sementara inferensi meregangkan
seluruh frame. Resize biasa mempertahankan seluruh konten di mana-mana; satu-satunya
biaya adalah sedikit distorsi rasio aspek, yang ditoleransi model dengan baik
untuk tugas ini.

---

## Pengaturan GPU

### Jalur A - Linux / WSL2 (TF 2.x modern, CUDA 12.x) - direkomendasikan

```bash
python -m venv hsg_env
source hsg_env/bin/activate

pip install --upgrade pip
pip install "tensorflow[and-cuda]"        # menginstal pustaka CUDA yang kompatibel secara otomatis
pip install "numpy<2"
pip install "opencv-python-headless==4.9.0.80"
pip install albumentations scikit-learn scikit-image scipy pandas matplotlib tqdm jupyter

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Jalur B - Windows native GPU (TF < 2.11, CUDA 11.x)

TensorFlow > 2.10 tidak mendukung GPU native di Windows; gunakan 2.10 dengan CUDA 11.2.

```bash
conda create --name hsg python=3.9
conda activate hsg
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

pip install --upgrade pip
pip install "tensorflow<2.11"
pip install "numpy<2"
pip install "opencv-python-headless==4.9.0.80"
pip install albumentations scikit-learn scikit-image scipy pandas matplotlib tqdm jupyter

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Output yang diharapkan: `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`

Kedua keluarga TF bekerja: API `mixed_precision` modern dicoba terlebih dahulu dan
API experimental lama digunakan sebagai fallback.

---

## Konfigurasi pelatihan (RTX 4050, 6 GB)

Dikonfigurasi di bagian atas `train.ipynb`:

| Pengaturan | Nilai |
|---|---|
| Ukuran batch | 2 (batas aman untuk ResNet50 + ASPP pada 512x512 di 6 GB) |
| Learning rate (warmup) | 1e-4 |
| Learning rate (fine-tune) | 1e-5 (lihat "Warmup dua fase" di bawah) |
| Epoch | 70 maks, EarlyStopping(patience=10, restore_best_weights) |
| Steps per epoch | 1500 (lihat "Steps per epoch" di bawah) |
| Optimizer | AdamW, weight_decay=1e-4, clipnorm=1.0 |
| Jadwal LR | ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-7) |
| Mixed precision | mixed_float16 (Tensor Cores) |
| Memori GPU | memory growth diaktifkan (tanpa pra-alokasi VRAM penuh) |

Ukuran batch 2 disengaja: backward pass untuk decoder penuh pada 512x512
tidak akan muat di 6 GB pada batch 4. Naikkan `BATCH_SIZE` ke 4 hanya pada
GPU 8 GB atau lebih besar. Sel pelatihan hanya melatih ketika `files/model.h5`
belum ada (`TRAIN = not os.path.exists(MODEL_PATH)`); **hapus `files/model.h5`
(dan `files/data.csv`) untuk memaksa pelatihan ulang** — menyetel `TRAIN = True`
secara manual tidak cukup karena baris tersebut menimpanya lagi. `train.ipynb`
juga punya toggle `QUICK_TEST` (1 epoch, sedikit batch, tanpa warmup) untuk
memverifikasi pipeline dalam hitungan menit sebelum menjalankan training penuh.

### Steps per epoch

Dengan augmentasi 8x penuh (~40.888 pasangan), satu epoch lengkap di RTX 4050
memakan waktu ~1,7 jam — untuk 50 epoch itu bisa 1-3+ hari. `train.ipynb`
menyediakan `STEPS_PER_EPOCH` (default 1500, ~9-9,3 menit/epoch termasuk
validasi penuh, dengan buffer untuk throttling termal pada run panjang) yang
membatasi setiap epoch pada sejumlah batch tersebut; dataset train
di-`.repeat()` sehingga tidak ada sampel yang dibuang, hanya tersebar pada
lebih banyak epoch. Dengan 70 epoch maks, ini muat dalam anggaran ~11 jam.
Setel ke `None` untuk kembali ke "satu epoch = seluruh dataset sekali".

### Warmup dua fase (default aktif: 3 epoch)

`train.ipynb` menyediakan `FREEZE_BACKBONE_EPOCHS` (default **3**). Selama
epoch-epoch ini, hanya decoder/ASPP yang dilatih (dengan `LR`, 1e-4) dengan
backbone ResNet50 dibekukan; setelah itu backbone dicairkan dan seluruh
jaringan di-fine-tune dengan learning rate **terpisah dan jauh lebih kecil**,
`FINETUNE_LR` (default 1e-5). BatchNormalization backbone tetap dibekukan
sepanjang proses. Dua hal ini bersama-sama menstabilkan pelatihan pada ukuran
batch 2 di bawah `mixed_float16`: memakai `LR` warmup yang sama pada fase
fine-tune bisa membuat backbone pretrained kolaps total begitu dicairkan
(val_iou anjlok ke ~0 dan tidak pulih, bukan cuma satu batch buruk yang
pulih sendiri) - lihat "Pemecahan masalah". Setel `FREEZE_BACKBONE_EPOCHS`
ke `0` untuk kembali ke training satu fase (tidak disarankan).

---

## Metrik

Metrik pelatihan adalah TensorFlow murni dan berjalan di GPU setiap batch:
koefisien Dice, IoU keras (Jaccard setelah thresholding pada 0.5), Recall, dan
Precision. Objektif adalah Dice loss + Focal loss; Focal (gamma=2, alpha=0.25)
memusatkan gradien pada piksel sulit dan piksel batas, yang penting karena
sebagian besar piksel adalah latar belakang.

Evaluasi tambahan melaporkan, per gambar: Accuracy, F1, dan **Boundary IoU**
(IoU yang dibatasi pada pita batas mask). Boundary IoU adalah fungsi NumPy/SciPy
yang hanya digunakan dalam evaluasi; terlalu lambat untuk dihitung setiap batch
pelatihan.

---

## Visualisasi

Semua figur ditulis ke `plots/` dan ditampilkan inline di notebook.

| File | Tahap | Konten |
|---|---|---|
| `01_file_size_distribution.png` | Persiapan | Distribusi ukuran file seluruh dataset |
| `02_selected_samples.png` | Persiapan | Grid sampel gambar dari dataset |
| `02a_data_exploration.png` | EDA | Scatter resolusi, histogram aspek, ukuran file, ringkasan |
| `02b_sample_data.png` | EDA | Grid original / mask / overlay |
| `02c_foreground_ratio.png` | EDA | Distribusi rasio piksel foreground |
| `02d_channel_intensity.png` | EDA | Distribusi intensitas per kanal (BGR) |
| `03_augmentation_showcase.png` | Augmentasi | Satu gambar dengan semua 8 varian (gambar + mask) |
| `04_dataset_split.png` | Augmentasi | Pie train/test + bar sebelum/sesudah augmentasi |
| `04b_pipeline_batch.png` | Pipeline | Batch tf.data yang didekode (sanity check bentuk + skala) |
| `05_training_history.png` | Pelatihan | Loss, Dice, IoU, Recall, dan jadwal LR |
| `06_evaluation_metrics.png` | Evaluasi | Bar chart metrik rata-rata |
| `07_prediction_samples.png` | Evaluasi | Baris original / GT / prediksi / masked |
| `08_metrics_distribution.png` | Evaluasi | Box plot metrik per gambar |
| `08b_best_worst.png` | Evaluasi | Prediksi terbaik dan terburuk berdasarkan Jaccard |
| `09_inference_results.png` | Inferensi | Grid input / masked untuk gambar baru |

---

## Augmentasi

Delapan varian ditulis per gambar pelatihan (yang pertama adalah original tanpa
modifikasi). Transformasi spasial diterapkan secara identik pada gambar dan
mask-nya; mask di-binarisasi ulang ke 0/255 setelah setiap transformasi.

1. Original
2. Flip Horizontal
3. Grayscale (grey 3-kanal)
4. Channel Shuffle
5. Coarse Dropout (3-10 lubang persegi panjang)
6. Rotasi (+/-45 derajat)
7. Elastic Transform (alpha=120, sigma=6)
8. Random Brightness/Contrast (+/-30%)

Daftar augmentasi ditulis untuk albumentations 2.x (signature `CoarseDropout`
terkini, tanpa `alpha_affine` pada `ElasticTransform`).

---

## Pemecahan masalah

**`%run metrics.ipynb` tidak ditemukan** - jalankan notebook dari direktori
project agar notebook sibling berada di path, atau buka Jupyter dengan folder
project sebagai direktori kerja.

**Konflik NumPy / OpenCV** (`opencv-python requires numpy>=2`):
```bash
pip uninstall -y opencv-python opencv-python-headless
pip install "opencv-python-headless==4.9.0.80"
```

**Kehabisan memori saat pelatihan** - pastikan `BATCH_SIZE = 2`, tutup proses
GPU lain, dan biarkan memory growth aktif (sudah disetel di `train.ipynb`).

**`model.compile` gagal dengan "Cannot take the length of shape with unknown
rank"** - ini disebabkan oleh metrik IoU berbasis `tf.numpy_function` yang lama
dan sudah diperbaiki: metrik pelatihan sekarang menggunakan TensorFlow murni.

**Paket yang hilang**:
```bash
pip install scikit-learn scipy
```

**`loss`/`dice_coef` jadi `nan` saat training, dan tidak ada checkpoint yang
tersimpan di `files/model.h5`** - ini risiko nyata di bawah `mixed_float16`:
satu batch dengan forward pass yang meledak (NaN) cukup untuk meracuni
rata-rata berjalan `loss`/`dice_coef` untuk *sisa* epoch itu, sehingga
`val_loss` jadi NaN dan `ModelCheckpoint` tidak akan pernah melihat
"peningkatan". `iou`/`recall`/`precision` tetap masuk akal karena metrik ini
men-threshold prediksi dulu (`NaN > 0.5` selalu `False` di TensorFlow),
sehingga satu batch NaN hanya dihitung salah, bukan meracuni epoch. Mitigasi:
`make_callbacks` memantau `val_iou` (mode="max"), bukan `val_loss`, jadi
checkpoint tetap tersimpan meski sesekali ada batch NaN.

**`val_iou` anjlok ke ~0 dan tidak pulih tepat saat backbone dicairkan (akhir
fase warmup)** - ini bukan satu batch NaN yang bisa pulih sendiri, tapi model
yang benar-benar kolaps: backbone ResNet50 pretrained menerima update sebesar
`LR` warmup begitu dicairkan, cukup besar untuk merusak bobot pretrained-nya.
Mitigasi: fase fine-tune memakai `FINETUNE_LR` terpisah (default 1e-5, 10x
lebih kecil dari `LR` warmup). Efek samping dari kolaps ini yang juga sudah
diperbaiki: `ModelCheckpoint` fase 2 sebelumnya mulai menghitung "terbaik"
dari `-inf` lagi, sehingga val_iou hasil kolaps (mis. 0.0) dianggap
"peningkatan" dari `-inf` dan **menimpa checkpoint bagus dari fase warmup**.
`run_training` sekarang men-seed `initial_value_threshold` fase 2 dengan
val_iou terbaik dari fase 1, jadi checkpoint fase 1 tidak akan pernah
tertimpa oleh checkpoint fase 2 yang lebih buruk.

---

## Catatan perangkat keras

| Sumber daya | Rekomendasi |
|---|---|
| VRAM | 6 GB (batch=2 dengan mixed precision); 8 GB+ untuk mencoba batch=4 |
| RAM | 16 GB direkomendasikan untuk dataset augmentasi |
| Penyimpanan | ~5 GB untuk dataset augmentasi + checkpoint |

---

## Pengakuan

- Dataset: People Segmentation Dataset oleh [Supervisely](https://supervise.ly/)
- Arsitektur: DeepLabV3+ - Chen et al., Google Research
- Framework: TensorFlow / Keras

---

## Lisensi

Kode di repo ini dilisensikan MIT - lihat [LICENSE](LICENSE). Dataset People
Segmentation (Supervisely) memiliki ketentuan lisensinya sendiri; lisensi MIT
di sini hanya mencakup kode, bukan dataset.
