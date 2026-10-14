# Optimasi Hybrid Ensemble dengan Cost-Sensitive Learning untuk Deteksi Dini Prediabetes dan Diabetes Melitus Tipe 2 pada Data Tidak Seimbang

Tugas Akhir D3 Teknologi Informasi (Kelompok TA_06).

> **Disclaimer:** proyek ini adalah prototipe skrining awal berbasis data gaya hidup,
> **bukan alat diagnosis medis**, dan belum melalui uji klinis.

## Ringkasan
Model stacking (XGBoost + Random Forest sebagai model dasar, Logistic Regression sebagai
meta-learner) dengan cost-sensitive learning pada data BRFSS 2015 tiga kelas
(tidak diabetes, prediabetes, diabetes). Kriteria keberhasilan utama adalah recall kelas
prediabetes, dilengkapi threshold tuning per kelas, kalibrasi probabilitas, dan evaluasi
pada subset fitur yang lebih dapat diterapkan di Indonesia.

## Arsitektur
Flutter (Android) -> REST API (FastAPI) -> model stacking (`.joblib`) -> hasil skrining

## Struktur
| Folder | Isi |
|---|---|
| `ml/` | pipeline pelatihan dan evaluasi, notebook EDA/SHAP, hasil eksperimen |
| `backend/` | FastAPI sebagai pusat inferensi |
| `mobile/` | aplikasi Flutter |
| `data/` | petunjuk mengunduh dataset (CSV tidak di-commit) |
| `docs/` | proposal, mockup, diagram arsitektur |

## Menjalankan pipeline ML
```
pip install -r requirements.txt
python ml/pipeline_prediabetes.py --data data/diabetes_012_health_indicators_BRFSS2015.csv --out ml/hasil
```

## Hasil evaluasi
_Diisi setelah pipeline dijalankan pada data asli (lihat `ml/hasil/results.csv`)._

## Keterbatasan
- BRFSS berasal dari survei di Amerika Serikat dan labelnya *self-reported*.
- Model belum divalidasi pada populasi Indonesia (Sumatera Utara/Toba); ini agenda lanjutan.

## Tim
| NIM | Nama |
|---|---|
| 42324019 | Joy Marthin Aruan |
| 42324030 | Rido Jeremi Siagian |
| 42324053 | Monica Sitanggang |

Pembimbing: Tegar Arifin Prasetyo, S.Si, M.Si; Tiurma Lumban Gaol, S.P., M.P.
