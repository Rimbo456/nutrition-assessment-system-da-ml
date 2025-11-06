Preprocessing foodseg103 dataset

Files added/updated:
- `preprocess_foodseg103.py`: main preprocessing script. Usage:
  - python project\preprocess_foodseg103.py --src <path-to-foodseg103> --dst <output-path> --size 512 512
  - Optional flags: --val-size, --seed, --debug, --debug-dir
  - When --debug is set, failed bitmap decodes (or base64 blobs) will be saved into <dst>/debug_bitmaps (or --debug-dir).

- `validate_masks.py`: small helper to compute how many masks are non-empty per split.

Notes & recommendations
- Create a virtual environment and install dependencies:
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -U pip
  pip install numpy opencv-python tqdm scikit-learn

- Run preprocessing locally (I will not run it unless you ask):
  python project\preprocess_foodseg103.py --src "D:\Dev\University\KLTN\foodseg103" --dst "D:\Dev\University\KLTN\project\data\foodseg103" --size 512 512

- To enable debug artifacts (helpful if masks are empty), add --debug. Example:
  python project\preprocess_foodseg103.py --debug --debug-dir "D:\temp\fs_debug"

- After running, run validation to get a quick summary:
  python project\validate_masks.py

If you want, I can further:
- Add a "verbose" logging mode to the preprocessor to print per-file reasons for skip.
- Add conversion scripts to create PyTorch Dataset or TFRecords.
- Add unit tests that validate small sample annotations.

Tell me which of these you want next and I will implement the files (without running anything).