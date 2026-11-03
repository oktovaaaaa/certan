"""Jalankan sebelum ujian dengan koneksi internet. Tidak membuat data sintetis."""
from pathlib import Path
import urllib.request
import csv, io, hashlib
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
path = Path(__file__).resolve().parent / 'datasets' / 'pima-indians-diabetes.data.csv'
raw = urllib.request.urlopen(url, timeout=30).read()
rows = list(csv.reader(io.StringIO(raw.decode('utf-8'))))
assert len(rows) == 768 and all(len(r) == 9 for r in rows), 'Ukuran/skema tidak sesuai'
values = [[float(v) for v in r] for r in rows]
assert sum(r[-1] == 0 for r in values) == 500
assert sum(r[-1] == 1 for r in values) == 268
assert values[0] == [6,148,72,35,0,33.6,0.627,50,1], 'Baris pertama berbeda'
path.parent.mkdir(exist_ok=True)
path.write_bytes(raw)
print('Tersimpan:', path)
print('Shape: 768 x 9, tanpa header. Kelas 0: 500; kelas 1: 268.')
print('SHA256 berkas yang diunduh:', hashlib.sha256(raw).hexdigest())
