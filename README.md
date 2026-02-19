# Diabetes Classification (MLflow + DVC + DagsHub)

End-to-end machine learning project to **classify diabetes outcomes** using tabular health features.  
This repo demonstrates a clean MLOps workflow with:

- **DVC** for data/versioning and reproducible pipelines  
- **MLflow** for experiment tracking, model registry-ready artifacts  
- **DagsHub** as the remote hub for **Git + DVC storage + MLflow tracking UI**

---

## Project Highlights

✅ Reproducible training using a DVC pipeline  
✅ Experiment tracking (params, metrics, artifacts) in MLflow  
✅ Remote tracking and storage via DagsHub  
✅ Model evaluation reports + saved model artifacts  
✅ Easy to reproduce on any machine

---

## Tech Stack

- Python (scikit-learn)
- MLflow (tracking + artifacts)
- DVC (data & pipeline)
- DagsHub (remote storage + MLflow UI)
- Pandas, NumPy

---

## ⚙️ Setup & Run Locally
### 1️⃣ Clone
```bash
git clone https://github.com/saadtariq-ds/diabetes-classification.git
cd diabetes-classification
```

### 2️⃣ Create virtual environment (recommended)
```bash
python -m venv ven
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

## For Adding DVC Stages
bash```
dvc stage add -n preprocess \
	-p preprocess.input_dir,preprocess.output_dir \
	-d src/preprocess.py -d data/raw/data.csv \
	-o data/processed/preprocessed_data.csv \
	python src/preprocess.py
```

dvc stage add -n train \
	-p train.data_dir,train.model_dir,train.random_state \
	-d src/train.py -d data/processed/preprocessed_data.csv \
	-o models/random_forest.pkl \
	python src/train.py

dvc stage add -n evaluate \
	-d src/evaluate.py -d models/model.pkl -d data/raw/data.csv \
	python src/evaluate.py