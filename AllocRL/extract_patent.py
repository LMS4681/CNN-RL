# -*- coding: utf-8 -*-
"""Read patent.docx from AllocRL folder (copy of the original)."""
import zipfile
import xml.etree.ElementTree as ET
import pathlib

# Try AllocRL/patent.docx first (likely a copy of the original)
paths_to_try = [
    pathlib.Path(r"D:\Sub\Allocation\AllocBase\AllocRL\patent.docx"),
]

target = None
for p in paths_to_try:
    if p.exists():
        target = p
        break

print(f"Using: {target}")

if target:
    with zipfile.ZipFile(target, 'r') as z:
        with z.open('word/document.xml') as doc_xml:
            tree = ET.parse(doc_xml)
    
    root = tree.getroot()
    
    out_path = pathlib.Path(r"D:\Sub\Allocation\AllocBase\data\patent_rl_text.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(str(out_path), 'w', encoding='utf-8') as f:
        para_idx = 0
        for para in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
            texts = []
            for run in para.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
                if run.text:
                    texts.append(run.text)
            line = ''.join(texts)
            if line.strip():
                f.write(f"{para_idx:>4}: {line}\n")
            para_idx += 1
    
    print(f"Done. Written to {out_path}")
else:
    print("No file found!")
