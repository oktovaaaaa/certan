# Paket latihan UTS Praktikum CERTAN 2025

## Status jujur
Solusi mencakup seluruh perintah Soal 1, Soal 2, serta dua varian Soal 3.
Notebook belum dijalankan pada dataset ujian. Tidak ada metrik/grafik hasil ujian yang dikarang.
Kode telah diperiksa sintaks Python dan struktur notebook, bukan diuji end-to-end.
Sandbox tidak memiliki scikit-learn, nbformat, dan nbclient. Unduhan internet dari sandbox gagal karena DNS; sumber web tetap dapat dibaca untuk memeriksa dataset Pima.

## Dataset
1. users_data_cleaned.pickle: belum ditemukan sumber publik terverifikasi; minta berkas asli e-course/pengampu. Wajib memuat setidaknya age, number_transactions, date_joined, total_amount_usd, job, marital. Jangan membuka pickle dari sumber tidak dipercaya.
2. Sales_Data.csv: belum ditemukan versi terverifikasi dengan Biaya Promo/Promosi dan Nilai Penjualan. File bernama sama dari sumber lain tidak otomatis benar.
3. pima-indians-diabetes.data.csv: sumber publik cocok dengan nama dan skema soal. 768 baris, 8 fitur + class, tanpa header. Kelas 0=500, kelas 1=268. Belum dibundel dalam paket ini. Jalankan `python unduh_pima.py` dengan internet sebelum ujian atau unduh manual:
https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
Dokumentasi:
https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names
Sumber ini sesuai dataset standar yang dimaksud, tetapi tanpa file resmi tidak dapat membuktikan identitas byte-per-byte dengan salinan e-course.
4. konsumen.csv: belum ditemukan versi asli terverifikasi berkolom No, Gaji, Pengeluaran. Mall_Customers bukan pengganti identik; spending score bukan nilai pengeluaran uang.

Tidak ada data sintetis/pengganti yang diam-diam dipakai.

## Cara pakai
- Ekstrak ZIP. Letakkan data asli dalam datasets/.
- Instal dependensi: python -m pip install -r requirements.txt
- Unduh Pima jika diperlukan: python unduh_pima.py
- Buka Pembahasan_UTS_Praktikum_CERTAN.ipynb dengan Jupyter.
- Jalankan persiapan dan bagian yang datanya tersedia. Untuk NIM ganjil gunakan K-NN; untuk NIM genap gunakan K-Means.
- File solusi_praktikum.py memuat kode yang sama dengan pemisah # %% agar dapat dibuka per sel di VS Code. Jangan Run All jika data belum lengkap.
- Output PNG/CSV ditulis ke hasil/ setelah dijalankan.
- PDF Panduan_Praktikum_CERTAN.pdf hanya panduan, bukan hasil running.
- Setelah berhasil running, simpan notebook ber-output dan ekspor PDF; kemas sebagai NIM_UTS_praktikum_Certan.zip sesuai aturan.

## Keputusan penting
- Soal 2 ambigu. Default mengikuti butir grafik: X=biaya promosi, y=penjualan. Bisa dibalik melalui ARAH setelah konfirmasi, lalu latih ulang.
- Split regresi 80/20; KNN 85/15. Seed 42, KNN stratified. Hasil tidak harus sama bila seed/split/preprocessing berbeda.
- KNN baseline memakai K=5, metric=euclidean. Versi imputasi+scaling dipisahkan.
- Grafik K 1..10 mencantumkan test accuracy untuk rubrik dan CV pada training untuk pembelajaran pemilihan K.
- KMeans memakai 2 cluster, n_init=10, random_state=42, fitur Gaji/Pengeluaran pada satuan asli.
- date_joined diagregasikan per tanggal dengan jumlah total_amount_usd, dilabeli sebagai cohort tanggal bergabung, bukan tanggal transaksi.
- Histogram/statistik/korelasi hanya fitur numerik. Heatmap memakai korelasi Pearson, bukan prediksi masa depan.

## Aturan ujian
Dokumen 2025 melarang internet dan AI selama ujian. Paket ini untuk latihan sebelum ujian. Pastikan pengampu mengizinkan bahan yang disusun menggunakan AI; jangan mengasumsikan semua file lokal otomatis boleh dipakai.
