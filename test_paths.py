"""Проверка авто-путей (fallback на /data/inputs)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = HERE
files = {
    'M5': os.path.join(DATA, "EURUSD_M5_202504180935_202608271920.csv"),
    'H1': os.path.join(DATA, "EURUSD_H1_202401020000_202608271900.csv"),
    'H4': os.path.join(DATA, "EURUSD_H4_202401020000_202608271600.csv"),
    'D1': os.path.join(DATA, "EURUSD_Daily_202401020000_202608270000.csv"),
}
print("HERE =", HERE)
print("M5 exists рядом со скриптом?", os.path.exists(files['M5']))
if not os.path.exists(files['M5']):
    DATA = "/data/inputs"
    for k in files:
        files[k] = os.path.join(DATA, os.path.basename(files[k]))
    print("FALLBACK ->", DATA)
for k, p in files.items():
    print(f"  {k}: {os.path.exists(p)}  {p}")
