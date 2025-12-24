# Workflow-CI

Repository untuk implementasi Continuous Integration (CI) dan model training dengan MLflow.

## Author
**Nama:** Syifa Fauziah  
**Course:** Membangun Sistem Machine Learning - Dicoding Indonesia

## Struktur Repository

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ml_training.yml        # GitHub Actions CI/CD pipeline
├── MLProject/
│   ├── MLProject                  # MLflow Project definition
│   ├── modelling.py               # Baseline model training (autolog)
│   ├── modelling_tuning.py        # Hyperparameter tuning (manual log)
│   ├── conda.yaml                 # Conda environment specification
│   ├── winequality_preprocessing/ # Preprocessed dataset
│   └── DockerHub.txt              # Link Docker Hub repository
└── README.md
```

## GitHub Actions Workflow

Workflow akan **otomatis berjalan** ketika:
- Push ke branch `main` atau `master`
- Pull request ke branch `main` atau `master`
- Manual trigger via "Run workflow"

### Cara Trigger Manual:
1. Buka tab **Actions** di repository
2. Pilih **ML Training Pipeline**
3. Klik **Run workflow**

### Workflow Steps:
1. **Checkout** - Clone repository
2. **Setup Python** - Install Python 3.9
3. **Install Dependencies** - Install required packages
4. **Fetch and Prepare Data** - Download dan preprocess data dari UCI
5. **Train Model with MLflow** - Training dengan autolog
6. **Upload Artifacts** - Simpan hasil training

## MLflow Project Entry Points

### Main (Default)
```bash
mlflow run MLProject -P data_dir=winequality_preprocessing
```

### Local Run
```bash
cd MLProject
python modelling.py --data-dir winequality_preprocessing
```

## Links
- **GitHub:** https://github.com/syfauziah/Workflow-CI
- **Eksperimen:** https://github.com/syfauziah/Eksperimen_SML_Syifa_Fauziah
- **Docker Hub:** https://hub.docker.com/r/syfauziah/wine-quality-mlops
- **DagsHub:** https://dagshub.com/syfauziah/wine-quality-mlops

## License
Educational use - Dicoding Indonesia
