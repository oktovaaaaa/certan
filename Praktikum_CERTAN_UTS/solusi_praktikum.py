# %% [markdown]
# Solusi latihan; lihat README untuk status data dan batas verifikasi.

# %% [markdown]
# # Pembahasan UTS Praktikum CERTAN 2025 — Bahan Latihan
# 
# Mencakup Soal 1, Soal 2, serta kedua varian Soal 3 (NIM ganjil dan genap).
# 
# **STATUS: template solusi; belum dijalankan pada dataset ujian.** Tiga dataset belum tersedia dan belum terverifikasi publik. Dataset Pima teridentifikasi di sumber publik yang sesuai, tetapi belum tersalin ke paket ini karena koneksi unduh sandbox gagal. Tidak ada hasil model, metrik, maupun grafik yang dikarang.
# 
# Gunakan untuk belajar sebelum ujian. Dokumen ujian 2025 melarang internet dan AI saat pelaksanaan; ikuti aturan pengampu tentang bahan yang boleh dibawa, termasuk bahan yang disusun dengan bantuan AI.
# 
# ## Dataset yang diperlukan
# 
# | Bagian | Nama berkas | Status |
# |---|---|---|
# | Soal 1 | users_data_cleaned.pickle | Memerlukan berkas asli dari e-course/pengampu |
# | Soal 2 | Sales_Data.csv | Memerlukan berkas asli; nama sama tidak membuktikan isi sama |
# | Soal 3 ganjil | pima-indians-diabetes.data.csv | Sumber publik cocok, 768 baris x 9 kolom, tanpa header |
# | Soal 3 genap | konsumen.csv | Memerlukan berkas asli berkolom No, Gaji, Pengeluaran |
# 
# Simpan berkas dalam folder `datasets/` di sebelah notebook. Setiap bagian dapat dijalankan terpisah setelah sel persiapan. Jangan mengeksekusi pickle dari sumber yang tidak dipercaya: format pickle dapat menjalankan kode.
# 
# **Pima:** https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
# 
# **Dokumentasi Pima:** https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names
# 
# Unduh sebelum ujian dengan `python unduh_pima.py` atau melalui browser, bukan saat ujian.

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path('datasets')  # ubah jika data disimpan di lokasi lain
OUT_DIR = Path('hasil')
OUT_DIR.mkdir(exist_ok=True)
SEED = 42
sns.set_theme(style='whitegrid')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 15)

def data_path(name):
    path = DATA_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f'{path} belum ada. Letakkan berkas asli pada folder datasets/.')
    return path

def clean_headers(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.replace('\ufeff', '', regex=False)
    if df.columns.duplicated().any():
        raise ValueError('Nama kolom duplikat; periksa berkas asli.')
    return df

def require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f'Kolom tidak ditemukan: {missing}. Kolom aktual: {df.columns.tolist()}')

def numeric_checked(df, cols):
    out = df.copy()
    for c in cols:
        converted = pd.to_numeric(out[c], errors='coerce')
        invalid = out[c].notna() & converted.isna()
        if invalid.any():
            raise ValueError(f'{c}: ada teks nonnumerik. Periksa separator/desimal; jangan menebak satuan.')
        out[c] = converted
    if not np.isfinite(out[cols].to_numpy(dtype=float)).all():
        raise ValueError('Ada missing/infinite pada fitur numerik. Tentukan penanganan berdasarkan data asli.')
    return out

def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT_DIR / name, dpi=150, bbox_inches='tight')
    plt.show()

print('Persiapan selesai. Folder data:', DATA_DIR.resolve())

# %% [markdown]
# ## Soal 1 — Eksplorasi dan visualisasi (30 poin)
# 
# Setiap sel berikut dipetakan ke nomor perintah pada soal. Nama `number_transaction` pada butir 9 kemungkinan salah ketik dari `number_transactions`; kode menangani kedua nama dan menolak jika keduanya hadir sekaligus tanpa kepastian.
# 
# ### 1. Load dan preview data
# Jalankan sel ini hanya untuk pickle resmi yang dipercaya.

# %%
users = pd.read_pickle(data_path('users_data_cleaned.pickle'))
if not isinstance(users, pd.DataFrame):
    raise TypeError('Objek pickle bukan DataFrame. Periksa struktur berkas resmi.')
users = clean_headers(users)
if 'number_transaction' in users.columns and 'number_transactions' not in users.columns:
    users = users.rename(columns={'number_transaction': 'number_transactions'})
elif 'number_transaction' in users.columns and 'number_transactions' in users.columns:
    raise ValueError('Ada dua kolom transaksi; konfirmasikan kolom yang dimaksud.')
require_columns(users, ['age', 'number_transactions', 'date_joined', 'total_amount_usd', 'job', 'marital'])
display(users.head())

# %% [markdown]
# ### 2. Shape data

# %%
print('Shape:', users.shape)
print('Jumlah baris:', users.shape[0])
print('Jumlah kolom:', users.shape[1])

# %% [markdown]
# ### 3. Tipe data

# %%
display(users.dtypes.rename('dtype').to_frame())
users.info()

# %% [markdown]
# ### 4. Korelasi — hanya kolom numerik

# %%
users_num = users.select_dtypes(include='number')
if users_num.empty:
    raise ValueError('Tidak ada kolom numerik. Periksa tipe data sebelum menghitung korelasi.')
corr = users_num.corr(method='pearson')
display(corr)
corr.to_csv(OUT_DIR / 'soal1_korelasi.csv')

# %% [markdown]
# ### 5. Statistik deskriptif numerik

# %%
display(users_num.describe())

# %% [markdown]
# ### 6. Histogram seluruh kolom numerik

# %%
users_num.hist(bins=20, figsize=(14, max(4, 3 * int(np.ceil(len(users_num.columns) / 3)))), layout=(-1, 3))
plt.suptitle('Distribusi seluruh kolom numerik', y=1.02)
savefig('soal1_histogram.png')

# %% [markdown]
# ### 7. Box plot age dan number_transactions — dipisah karena skala berbeda

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.boxplot(data=users, y='age', ax=axes[0])
sns.boxplot(data=users, y='number_transactions', ax=axes[1])
axes[0].set_title('Box plot age')
axes[1].set_title('Box plot number_transactions')
savefig('soal1_boxplot.png')

# %% [markdown]
# ### 8. Line plot date_joined terhadap total_amount_usd
# Tanggal dikonversi dan diurutkan. Jika beberapa pengguna memiliki tanggal bergabung yang sama, grafik berikut memakai **jumlah total_amount_usd per tanggal bergabung**. Ini profil cohort berdasarkan tanggal bergabung, bukan otomatis penjualan yang terjadi pada tanggal tersebut. Tanggal invalid dilaporkan, bukan disembunyikan.

# %%
line_data = users[['date_joined', 'total_amount_usd']].copy()
line_data['date_joined'] = pd.to_datetime(line_data['date_joined'], errors='coerce')
print('Tanggal tidak terbaca/kosong:', int(line_data['date_joined'].isna().sum()))
print('Nilai amount kosong:', int(line_data['total_amount_usd'].isna().sum()))
line_data = line_data.dropna(subset=['date_joined', 'total_amount_usd'])
line_data['tanggal'] = line_data['date_joined'].dt.normalize()
daily = line_data.groupby('tanggal', as_index=False)['total_amount_usd'].sum().sort_values('tanggal')
plt.figure(figsize=(12, 4))
sns.lineplot(data=daily, x='tanggal', y='total_amount_usd', estimator=None)
plt.title('Total amount pengguna menurut tanggal bergabung (agregasi jumlah)')
plt.xlabel('Tanggal bergabung')
plt.ylabel('Jumlah total_amount_usd')
plt.xticks(rotation=45)
savefig('soal1_lineplot.png')

# %% [markdown]
# ### 9. Scatter plot total_amount_usd dan number_transactions

# %%
plt.figure(figsize=(7, 5))
sns.scatterplot(data=users, x='number_transactions', y='total_amount_usd', alpha=0.65)
plt.title('Jumlah transaksi dan total amount')
savefig('soal1_scatter.png')

# %% [markdown]
# ### 10. Horizontal bar chart job — frekuensi pengguna

# %%
job_counts = users['job'].astype('string').fillna('(Missing)').value_counts().sort_values()
plt.figure(figsize=(9, max(4, 0.35 * len(job_counts))))
job_counts.plot.barh()
plt.xlabel('Jumlah pengguna')
plt.ylabel('Job')
plt.title('Frekuensi job')
savefig('soal1_job.png')

# %% [markdown]
# ### 11. Horizontal bar chart marital — frekuensi pengguna

# %%
marital_counts = users['marital'].astype('string').fillna('(Missing)').value_counts().sort_values()
plt.figure(figsize=(8, 4))
marital_counts.plot.barh()
plt.xlabel('Jumlah pengguna')
plt.ylabel('Marital')
plt.title('Frekuensi marital')
savefig('soal1_marital.png')

# %% [markdown]
# ### 12. Heatmap korelasi fitur — “future correlations” ditafsirkan sebagai “feature correlations”

# %%
plt.figure(figsize=(max(7, len(corr) * 0.9), max(5, len(corr) * 0.75)))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Korelasi Pearson fitur numerik')
savefig('soal1_heatmap.png')

# %% [markdown]
# ### 13. Identifikasi missing values

# %%
missing = pd.DataFrame({
    'jumlah_missing': users.isna().sum(),
    'persen_missing': users.isna().mean() * 100
}).sort_values('jumlah_missing', ascending=False)
display(missing)
print('Jumlah baris dengan minimal satu missing:', int(users.isna().any(axis=1).sum()))
missing.to_csv(OUT_DIR / 'soal1_missing.csv')

# %% [markdown]
# ## Soal 2 — LinearRegression (35 poin)
# 
# **Ambiguitas soal:** kalimat pembuka meminta prediksi biaya promosi, tetapi butir grafik f–g meminta penjualan berdasarkan biaya promosi. Implementasi utama mengikuti butir f–g: **X = Biaya Promo/Promosi, y = Nilai Penjualan**. Jika pengampu mengonfirmasi arah sebaliknya, ubah `ARAH` pada sel berikut, lalu jalankan ulang bagian Soal 2. Model terbalik harus dilatih ulang; bukan sekadar membalik persamaan.
# 
# ### 1a–b. Membaca file dan menampilkan variabel
# Nama header sebenarnya belum diketahui. Kode hanya menerima alias yang relevan dan berhenti jika ambigu. Periksa `sales.columns` dan atur nama secara eksplisit jika perlu. Jangan mengganti file ini dengan Salary_Data atau file penjualan lain yang tidak memiliki variabel sesuai.

# %%
sales = clean_headers(pd.read_csv(data_path('Sales_Data.csv'), sep=None, engine='python'))
display(sales.head())
print('Variabel:', sales.columns.tolist())
display(sales.dtypes)

def pick_column(columns, aliases):
    normalize = lambda s: ''.join(c for c in str(s).casefold() if c.isalnum())
    valid = {normalize(a) for a in aliases}
    matches = [c for c in columns if normalize(c) in valid]
    if len(matches) != 1:
        raise ValueError(f'Kolom ambigu/tidak ditemukan untuk {aliases}. Atur nama kolom sesuai berkas asli: {list(columns)}')
    return matches[0]

PROMO_COL = pick_column(sales.columns, ['Biaya Promo', 'Biaya Promosi'])
SALES_COL = pick_column(sales.columns, ['Nilai Penjualan', 'Penjualan'])
sales = numeric_checked(sales, [PROMO_COL, SALES_COL])
ARAH = 'penjualan_dari_promosi'
if ARAH == 'penjualan_dari_promosi':
    X_COL, Y_COL = PROMO_COL, SALES_COL
elif ARAH == 'promosi_dari_penjualan':
    X_COL, Y_COL = SALES_COL, PROMO_COL
else:
    raise ValueError('ARAH tidak dikenal')
X_sales = sales[[X_COL]]  # 2 dimensi untuk sklearn
Y_sales = sales[Y_COL]   # 1 dimensi untuk target
print('Prediktor X:', X_COL, '| Target y:', Y_COL)

# %% [markdown]
# ### 1c. Split training 80% dan testing 20%

# %%
Xtr_sales, Xte_sales, ytr_sales, yte_sales = train_test_split(
    X_sales, Y_sales, test_size=0.20, random_state=SEED
)
train_sales = Xtr_sales.join(ytr_sales)
test_sales = Xte_sales.join(yte_sales)
print('Train:', Xtr_sales.shape, ytr_sales.shape)
print('Test :', Xte_sales.shape, yte_sales.shape)
display(train_sales)
display(test_sales)
train_sales.to_csv(OUT_DIR / 'soal2_train.csv', index=True)
test_sales.to_csv(OUT_DIR / 'soal2_test.csv', index=True)

# %% [markdown]
# ### 1d–e. Membuat model, fit pada training, lalu predict testing

# %%
reg = LinearRegression()
reg.fit(Xtr_sales, ytr_sales)
yhat_sales = reg.predict(Xte_sales)
pred_sales = pd.DataFrame({
    X_COL: Xte_sales[X_COL],
    'aktual': yte_sales,
    'prediksi': yhat_sales,
    'residual': yte_sales - yhat_sales
})
with pd.option_context('display.max_rows', None):
    display(pred_sales)
pred_sales.to_csv(OUT_DIR / 'soal2_prediksi.csv', index=True)

# %% [markdown]
# ### 1f. Grafik hubungan nilai penjualan dan biaya promosi

# %%
plt.figure(figsize=(7, 5))
plt.scatter(sales[PROMO_COL], sales[SALES_COL], alpha=0.75)
plt.xlabel(PROMO_COL)
plt.ylabel(SALES_COL)
plt.title('Nilai Penjualan berdasarkan Biaya Promosi')
savefig('soal2_hubungan.png')

# %% [markdown]
# ### 1g. Garis prediksi pada training dan testing
# Garis diurutkan melalui grid X agar tidak zigzag. Kedua panel menggunakan **model yang sama**, dilatih hanya pada training; tidak fit ulang pada test.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, xx, yy, title in [
    (axes[0], Xtr_sales, ytr_sales, 'Training set'),
    (axes[1], Xte_sales, yte_sales, 'Test set')
]:
    ax.scatter(xx[X_COL], yy, color='royalblue', label='Aktual')
    grid = pd.DataFrame({X_COL: np.linspace(xx[X_COL].min(), xx[X_COL].max(), 100)})
    ax.plot(grid[X_COL], reg.predict(grid), color='crimson', label='Prediksi model training')
    ax.set(xlabel=X_COL, ylabel=Y_COL, title=title)
    ax.legend()
savefig('soal2_train_test_regresi.png')

# %% [markdown]
# ### 2a–c. Intercept, slope, dan R-square
# Untuk regresi sederhana: y_hat = intercept + slope * X. R-square test dapat negatif; nilainya bukan accuracy klasifikasi. Hasil dipengaruhi arah model, data, dan pembagian train/test.

# %%
print('Intercept:', reg.intercept_)
print('Slope:', reg.coef_[0])
print(f'Persamaan: {Y_COL} = {reg.intercept_:.6f} + ({reg.coef_[0]:.6f}) * {X_COL}')
print('R-square TRAIN:', reg.score(Xtr_sales, ytr_sales))
print('R-square TEST :', r2_score(yte_sales, yhat_sales) if len(yte_sales) >= 2 else 'Tidak terdefinisi: test kurang dari 2 sampel')
print('MSE TEST:', mean_squared_error(yte_sales, yhat_sales))

# %% [markdown]
# ## Soal 3 — NIM Ganjil: K-NN diabetes (35 poin)
# 
# ### 1–2. Membaca dataset dan menampilkan 8 baris teratas
# Pima versi publik tidak memiliki header. Jika berkas resmi ternyata memiliki header, ubah parameter pembacaan dengan sengaja; jangan membuang baris data pertama tanpa memeriksa.
# 
# `class=0`: negatif pada dataset, `class=1`: positif. Ini latihan klasifikasi historis, bukan alat diagnosis klinis.

# %%
pima_names = ['preg', 'glukosa', 'pres', 'skin', 'insulin', 'BMI', 'pedi', 'age', 'class']
pima = pd.read_csv(data_path('pima-indians-diabetes.data.csv'), header=None, names=pima_names)
pima = numeric_checked(pima, pima_names)
if not pima['class'].isin([0, 1]).all():
    raise ValueError('Label harus 0/1. Periksa apakah CSV memiliki header.')
pima['class'] = pima['class'].astype(int)
print('Shape:', pima.shape)
print('Distribusi kelas:', pima['class'].value_counts().sort_index().to_dict())
if pima.shape != (768, 9):
    print('PERHATIAN: ukuran berbeda dari versi publik 768 x 9. Periksa versi berkas.')
display(pima.head(8))

# %% [markdown]
# ### 3. Features dan label — 9 baris teratas

# %%
X_pima = pima.drop(columns='class')
y_pima = pima['class']
print('Features:')
display(X_pima.head(9))
print('Label:')
display(y_pima.head(9).to_frame())

# %% [markdown]
# ### 4. Split training 85% dan testing 15%
# `stratify` mempertahankan proporsi kelas sejauh dimungkinkan. Untuk 768 data: 652 train dan 116 test karena ukuran test dibulatkan ke atas oleh sklearn.

# %%
Xtr_pima, Xte_pima, ytr_pima, yte_pima = train_test_split(
    X_pima, y_pima, test_size=0.15, random_state=SEED, stratify=y_pima
)
print('X_train:', Xtr_pima.shape, '| y_train:', ytr_pima.shape)
print('X_test :', Xte_pima.shape, '| y_test :', yte_pima.shape)
display(Xtr_pima.join(ytr_pima))
display(Xte_pima.join(yte_pima))
Xtr_pima.join(ytr_pima).to_csv(OUT_DIR / 'soal3_ganjil_train.csv')
Xte_pima.join(yte_pima).to_csv(OUT_DIR / 'soal3_ganjil_test.csv')

# %% [markdown]
# ### 5–6. K=5, Euclidean distance, dan accuracy
# Baseline berikut mengikuti algoritma yang diminta tanpa menambahkan perubahan data diam-diam. K-NN sensitif terhadap skala dan nilai nol tak-valid pada beberapa pengukuran Pima. Versi preprocessing terpisah diberikan di bawah; metriknya tidak harus sama dengan baseline.

# %%
knn5 = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn5.fit(Xtr_pima, ytr_pima)
pred_pima = knn5.predict(Xte_pima)
acc_pima = accuracy_score(yte_pima, pred_pima)
print(f'Accuracy K=5 baseline: {acc_pima:.4f} ({acc_pima:.2%})')
print('Confusion matrix:')
print(confusion_matrix(yte_pima, pred_pima))
print(classification_report(yte_pima, pred_pima, target_names=['Negatif', 'Positif'], zero_division=0))
display(pd.DataFrame({'aktual': yte_pima, 'prediksi': pred_pima}))

# %% [markdown]
# ### 7. Grafik K terhadap accuracy, K=1 sampai 10
# Untuk memenuhi grafik pada soal, accuracy test ditampilkan untuk setiap K. **Jangan menganggap K terbaik pada kurva test menghasilkan estimasi generalisasi yang bebas bias**, karena test sudah dipakai membandingkan model. Kolom CV dihitung hanya di training dan dapat digunakan untuk memilih K dengan lebih tepat.

# %%
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
k_rows = []
for k in range(1, 11):
    m = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    cv_mean = cross_val_score(m, Xtr_pima, ytr_pima, cv=cv, scoring='accuracy').mean()
    m.fit(Xtr_pima, ytr_pima)
    k_rows.append({'k': k, 'accuracy_test': m.score(Xte_pima, yte_pima), 'accuracy_cv_train': cv_mean})
k_scores = pd.DataFrame(k_rows)
display(k_scores)
plt.figure(figsize=(8, 4))
plt.plot(k_scores['k'], k_scores['accuracy_test'], 'o-', label='Test (grafik sesuai soal)')
plt.plot(k_scores['k'], k_scores['accuracy_cv_train'], 's--', label='CV pada training')
plt.xticks(range(1, 11))
plt.xlabel('K (jumlah tetangga)')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.title('K-NN: K terhadap accuracy')
plt.legend()
savefig('soal3_ganjil_k_accuracy.png')
k_scores.to_csv(OUT_DIR / 'soal3_ganjil_k_accuracy.csv', index=False)
print('K terbaik menurut CV training:', int(k_scores.loc[k_scores['accuracy_cv_train'].idxmax(), 'k']))

# %% [markdown]
# ### Tambahan pembelajaran — imputasi dan scaling yang benar
# Nol pada `preg` dapat sah. Nol pada glukosa, tekanan darah, ketebalan kulit, insulin, dan BMI diperlakukan sebagai pengukuran hilang pada versi ini. Median dan scaling **fit hanya pada training** melalui pipeline. Jangan menyebut hasil ini identik dengan baseline.

# %%
zero_missing_cols = ['glukosa', 'pres', 'skin', 'insulin', 'BMI']
Xtr_clean = Xtr_pima.copy()
Xte_clean = Xte_pima.copy()
Xtr_clean[zero_missing_cols] = Xtr_clean[zero_missing_cols].replace(0, np.nan)
Xte_clean[zero_missing_cols] = Xte_clean[zero_missing_cols].replace(0, np.nan)
knn_preprocessed = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5, metric='euclidean'))
])
knn_preprocessed.fit(Xtr_clean, ytr_pima)
print('Accuracy K=5 setelah imputasi + scaling:', knn_preprocessed.score(Xte_clean, yte_pima))

# %% [markdown]
# ## Soal 3 — NIM Genap: K-Means konsumen (35 poin)
# 
# Implementasi utama menggunakan dua fitur asli, seperti soal. Jika satuan atau skala berbeda jauh, scaling layak dipertimbangkan, tetapi hasil cluster akan berubah; jangan menerapkannya diam-diam saat ingin mereplikasi hasil modul.
# 
# ### 1–2. Membaca konsumen.csv dan preview 7 baris

# %%
konsumen = clean_headers(pd.read_csv(data_path('konsumen.csv'), sep=None, engine='python'))
require_columns(konsumen, ['No', 'Gaji', 'Pengeluaran'])
display(konsumen.head(7))

# %% [markdown]
# ### 3–4. Hapus No dan tampilkan 9 baris

# %%
konsumen = konsumen.drop(columns='No')
display(konsumen.head(9))
konsumen = numeric_checked(konsumen, ['Gaji', 'Pengeluaran'])
X_konsumen = konsumen[['Gaji', 'Pengeluaran']]
if len(X_konsumen.drop_duplicates()) < 2:
    raise ValueError('Minimal perlu dua titik unik untuk membentuk dua cluster.')

# %% [markdown]
# ### 5. Grafik hubungan gaji dan pengeluaran

# %%
plt.figure(figsize=(7, 5))
plt.scatter(X_konsumen['Gaji'], X_konsumen['Pengeluaran'])
plt.xlabel('Gaji')
plt.ylabel('Pengeluaran')
plt.title('Hubungan Gaji dan Pengeluaran')
savefig('soal3_genap_hubungan.png')

# %% [markdown]
# ### 6–7. Dua cluster dan pusat cluster

# %%
kmeans = KMeans(n_clusters=2, random_state=SEED, n_init=10)
cluster_labels = kmeans.fit_predict(X_konsumen)
centers = pd.DataFrame(kmeans.cluster_centers_, columns=['Gaji', 'Pengeluaran'])
centers.index.name = 'Cluster'
print('Pusat cluster (satuan asli):')
display(centers)
print('Inertia/WCSS:', kmeans.inertia_)
hasil_cluster = X_konsumen.assign(Cluster=cluster_labels)
display(hasil_cluster)
hasil_cluster.to_csv(OUT_DIR / 'soal3_genap_cluster.csv', index=False)
centers.to_csv(OUT_DIR / 'soal3_genap_centroid.csv')

# %% [markdown]
# ### 8. Grafik konsumen dengan dua warna

# %%
colors = ['royalblue', 'darkorange']
plt.figure(figsize=(7, 5))
for c in range(2):
    points = X_konsumen.loc[cluster_labels == c]
    plt.scatter(points['Gaji'], points['Pengeluaran'], color=colors[c], label=f'Cluster {c}')
plt.xlabel('Gaji')
plt.ylabel('Pengeluaran')
plt.title('K-Means: dua cluster konsumen')
plt.legend()
savefig('soal3_genap_cluster.png')

# %% [markdown]
# ### 9. Grafik dua warna beserta pusat cluster

# %%
plt.figure(figsize=(7, 5))
for c in range(2):
    points = X_konsumen.loc[cluster_labels == c]
    plt.scatter(points['Gaji'], points['Pengeluaran'], color=colors[c], label=f'Cluster {c}')
plt.scatter(centers['Gaji'], centers['Pengeluaran'], marker='X', s=240,
            color='black', edgecolors='white', label='Pusat cluster')
plt.xlabel('Gaji')
plt.ylabel('Pengeluaran')
plt.title('K-Means: dua cluster dan centroid')
plt.legend()
savefig('soal3_genap_centroid.png')

# %% [markdown]
# ## Interpretasi dan pemeriksaan akhir
# 
# - Label cluster 0 dan 1 tidak mempunyai makna urutan; penomorannya dapat bertukar.
# - Intercept, slope, R-square, accuracy, centroid, dan grafik baru dapat disimpulkan setelah data asli dijalankan.
# - Korelasi tidak membuktikan sebab-akibat. ID numerik bukan otomatis fitur analitis bermakna.
# - Jangan menggunakan test set untuk melatih model maupun mempelajari parameter preprocessing.
# - Jika CSV menggunakan desimal koma atau pemisah ribuan, periksa berkas mentah dan sesuaikan `sep`, `decimal`, atau `thousands` secara eksplisit.
# 
# ## Menjalankan dan mengekspor
# 
# 1. Instal dependensi sebelum ujian dengan `python -m pip install -r requirements.txt` jika belum tersedia.
# 2. Buka notebook dari folder paket dan lengkapi berkas dalam `datasets/`.
# 3. Jalankan sel persiapan, Soal 1, Soal 2, lalu varian Soal 3 sesuai NIM. Kedua varian disediakan untuk latihan, bukan kewajiban mengumpulkan keduanya.
# 4. Jika ingin Run All, hapus/lewati varian Soal 3 yang tidak dikerjakan dan pastikan seluruh dataset yang dirujuk tersedia.
# 5. Periksa output, simpan notebook, lalu export HTML dan Print to PDF melalui browser atau menu ekspor Jupyter yang tersedia. PDF panduan dalam paket bukan PDF hasil running.
# 6. Kumpulkan notebook yang sudah berisi output dan PDF hasilnya dalam `NIM_UTS_praktikum_Certan.zip` sesuai aturan pengampu. Paket latihan ini belum merupakan berkas pengumpulan final.
# 
# Petunjuk umum halaman 1 menyebut output notebook yang diekspor PDF cukup, meskipun rubrik masih menyebut screenshot. Ikuti klarifikasi pengampu bila berbeda.
# 
# ## Latihan pemahaman
# 
# 1. Mengapa X regresi ditulis `df[[nama_kolom]]`, bukan `df[nama_kolom]`?
# 2. Mengapa kolom No tidak dipakai untuk K-Means?
# 3. Mengapa scaler tidak boleh di-fit pada seluruh data sebelum split?
# 4. Mengapa file Pima harus dibaca dengan `header=None`?
# 5. Apa perbedaan `fit`, `predict`, dan `fit_predict`?
