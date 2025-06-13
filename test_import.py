#!/usr/bin/env python3
import sys
import os

print("Current working directory:", os.getcwd())
print("Script location:", os.path.abspath(__file__))
print("Parent directory:", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add to path like the script does
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("\nPython path:")
for p in sys.path:
    print(f"  - {p}")

print("\nChecking if src exists:")
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
print(f"src path would be: {src_path}")
print(f"src exists: {os.path.exists(src_path)}")

if os.path.exists(src_path):
    print("\nContents of src:")
    for item in os.listdir(src_path):
        print(f"  - {item}")

# Try the import
try:
    from src.models.clinical_t5 import ClinicalT5Model, ClinicalT5Config
    print("\n✓ Import successful!")
except ImportError as e:
    print(f"\n✗ Import failed: {e}") 