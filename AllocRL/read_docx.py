# -*- coding: utf-8 -*-
import docx
import pathlib

path = pathlib.Path(r"D:\Sub\Allocation\AllocBase\AllocRL\patent.docx")
print(f"exists: {path.exists()}, size: {path.stat().st_size}")

# Try opening with zipfile first to check validity
import zipfile
try:
    with zipfile.ZipFile(str(path), 'r') as z:
        print(f"Valid ZIP with {len(z.namelist())} entries")
except Exception as e:
    print(f"Not valid ZIP: {e}")

doc = docx.Document(str(path))
for i, p in enumerate(doc.paragraphs):
    if p.text.strip():
        print(f"{i:>4}: [{p.style.name}] {p.text[:200]}")
