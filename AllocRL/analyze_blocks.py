"""블록 CSV 데이터 분포 분석 스크립트."""
import os, sys, csv
sys.path.insert(0, ".")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from collections import Counter
from datetime import datetime

csv_path = "data/블록데이터.csv"

COL_SHIP_NO      = 0
COL_BLOCK_NAME   = 1
COL_WORKSPACE    = 6
COL_LENGTH       = 9
COL_BREADTH      = 10
COL_HEIGHT       = 11
COL_WEIGHT       = 12
COL_PLACED_IN    = 15
COL_PLACED_OUT   = 16
COL_SCHEDULE_IN  = 17
COL_SCHEDULE_OUT = 18

def parse_float(v):
    try: return float(v.strip())
    except: return None

def parse_date(v):
    v = v.strip()
    if not v: return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try: return datetime.strptime(v, fmt).date()
        except: pass
    return None

lengths, breadths, heights, weights = [], [], [], []
durations = []
ws_codes = []
ship_nos = []
unplaced_count = 0
preplaced_count = 0

for enc in ("utf-8-sig", "cp949"):
    try:
        with open(csv_path, encoding=enc) as f:
            reader = csv.reader(f)
            header = next(reader)
            print(f"Header ({len(header)} cols): {header[:20]}")
            for row in reader:
                if len(row) < 19: continue
                l = parse_float(row[COL_LENGTH])
                b = parse_float(row[COL_BREADTH])
                h = parse_float(row[COL_HEIGHT])
                w = parse_float(row[COL_WEIGHT])
                ws = row[COL_WORKSPACE].strip()
                ship = row[COL_SHIP_NO].strip()

                if l is None or b is None or l <= 0 or b <= 0: continue

                si = parse_date(row[COL_SCHEDULE_IN])
                so = parse_date(row[COL_SCHEDULE_OUT])
                pi = parse_date(row[COL_PLACED_IN])
                po = parse_date(row[COL_PLACED_OUT])

                if ws and pi and po:
                    preplaced_count += 1
                    in_d, out_d = pi, po
                elif si and so:
                    unplaced_count += 1
                    in_d, out_d = si, so
                else:
                    continue

                lengths.append(l)
                breadths.append(b)
                if h: heights.append(h)
                if w: weights.append(w)
                if ws: ws_codes.append(ws)
                if ship: ship_nos.append(ship)

                dur = (out_d - in_d).days
                if dur > 0: durations.append(dur)
        break
    except UnicodeDecodeError:
        continue

def stats(name, arr):
    a = np.array(arr)
    pcts = np.percentile(a, [5, 25, 50, 75, 95])
    print(f"\n{name} (n={len(a)}):")
    print(f"  min={a.min():.2f}  max={a.max():.2f}  mean={a.mean():.2f}  std={a.std():.2f}")
    print(f"  P5={pcts[0]:.2f}  P25={pcts[1]:.2f}  P50={pcts[2]:.2f}  P75={pcts[3]:.2f}  P95={pcts[4]:.2f}")

print(f"\n총 블록: {preplaced_count + unplaced_count} (기배치={preplaced_count}, 미배치={unplaced_count})")
stats("Length (길이)", lengths)
stats("Breadth (폭)", breadths)
stats("Height (높이)", heights)
stats("Weight (중량)", weights)
stats("Duration (공기, 달력일)", durations)

print(f"\n작업장 코드 (상위 10):")
for code, cnt in Counter(ws_codes).most_common(10):
    print(f"  {code}: {cnt}")

print(f"\n호선 (상위 10):")
for ship, cnt in Counter(ship_nos).most_common(10):
    print(f"  {ship}: {cnt}")
