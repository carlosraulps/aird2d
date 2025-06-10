## AI Reverse-engineering & Discovery for 2D materials (AIRD-2D)

A project based in the Computational 2D Materials Database (C2DB) + custom code for training generative/discriminative ML models to predict and identify novel 2D materials from their structural and computed properties.

---

### 📁 Repository Layout

```
aird2d/
├── codes/            ← your Python modules and scripts  
├── databases/        ← raw or pre-built C2DB files (e.g. c2db.db or JSON tarball)  
├── data/             ← processed datasets, feature tables, train/test splits  
├── results/          ← figures, model checkpoints, prediction outputs  
├── references/       ← papers, notes, slides you’re building on  
├── requirements.txt  ← all pip-installable dependencies  
└── README.md         ← this file  
```

---

## ⚙️ Installation

1. **Clone** this repository

   ```bash
   git clone https://github.com/your-user/nano-ia.git
   cd aird2d
   ```

2. **Create & activate** a virtual environment (recommended)

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install** dependencies
   Before read [https://wiki.fysik.dtu.dk/ase/install.html#download-and-install](https://wiki.fysik.dtu.dk/ase/install.html#download-and-install) and [https://gpaw.readthedocs.io/install.html](https://gpaw.readthedocs.io/install.html)
   ```bash
   pip install -r requirements.txt
   ```

4. **Obtain** the C2DB data

   * **ASE database file**
     The dataset is provided upon request `c2db.db` (ASE-compatible SQLite) 
     and place it under `databases/`.
   * **JSON tarball** (alternative)
     Download the full JSON export, unpack under `databases/json/`.

---

## 🚀 Usage

### 1. Exploring the Database

In `codes/explore_c2db.py` you’ll find a example of use.

### 2. Feature Extraction

Scripts in `codes/features/` show how to:

* Compute atom-wise descriptors (e.g. Bader charges, Born charges)
* Aggregate per-material fingerprints (SOAP, ELEC, etc.)
* Export `data/features.csv`

### 3. Model Training

* **Generative model** (e.g. VAE/GAN):
  `codes/train_generative.py`
* **Discriminative model** (e.g. Random Forest, GNN):
  `codes/train_discriminative.py`

All entry-point scripts take a `--config` argument to point to your `.yaml` run settings.

### 4. Results & Visualization

Your trained models and evaluation logs go into `results/`. Example plotting scripts are in `codes/plotting/`.

---

## 📝 References

* **C2DB access & structure**
  Kristian S. Thygesen et al., “Recent progress of the Computational 2D Materials Database (C2DB)”, *2D Mater.* 8 (2021).
* **ASE Database Format**
  The `c2db.db` is an ASE SQLite database; see [https://wiki.fysik.dtu.dk/c2db/c2db.html#using-the-data](https://wiki.fysik.dtu.dk/c2db/c2db.html#using-the-data)

---

## 🛠️ Development

* **Code style:** PEP8, black formatting
* **Tests:** under `codes/tests/` (pytest)
* **CI:** GitHub Actions lint & simple smoke tests

---

## 🚩 Contributing

1. Create a branch `feature/…`
2. Add or update code/docs
3. Run tests: `pytest`
4. Submit a pull request

---

## 🗒️ License

[MIT License](./LICENSE)
Feel free to adapt or extend!

