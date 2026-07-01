# ICT-AMSOS: AI-Powered ICT Asset Management and Service Optimization System

A dual-model Machine Learning system that predicts ICT asset replacement priority and health scores to support data-driven procurement and maintenance planning.

## Architecture

```
Raw Data (CSV)
    │
    ▼
Feature Engineering
    ├── Equipment Age, Remaining Useful Life
    ├── CPU, RAM, Storage (regex extraction)
    ├── Repair Frequency, Latest Repair Info
    ├── Depreciation %, License Risk
    └── Asset Health Score (rule-based)
    │
    ▼
┌─────────────────────┐  ┌──────────────────────────────┐
│ RandomForest         │  │ RandomForest                 │
│ Classifier           │  │ Regressor                    │
│                     │  │                              │
│ Output:             │  │ Output:                      │
│ Replacement Priority│  │ Asset Health Score (0-100)   │
│ (Critical/High/     │  │                              │
│  Medium/Low)        │  │                              │
└─────────┬───────────┘  └──────────────┬───────────────┘
          │                             │
          ▼                             ▼
    ┌─────────────────────────────────────────┐
    │        Ensemble Decision Layer          │
    │  Replacement Score = blend of both      │
    │  Color-coded priority ranking           │
    │  Procurement recommendations            │
    └─────────────────────────────────────────┘
```

## Project Structure

```
ICT_AMSOS/
├── data/                          # Input CSV files
│   ├── inv_inventory.csv          # ICT asset inventory
│   ├── repairhistory.csv          # Equipment repair records
│   └── division_counts.csv        # Employee count per division
├── models/                        # Saved ML models (pickle)
│   ├── replacement_model.pkl      # Trained classifier
│   ├── health_score_model.pkl     # Trained regressor
│   ├── feature_columns.pkl        # Feature column names
│   └── label_encoder.pkl          # Target label encoder
├── outputs/                       # Exported results
│   ├── replacement_priority.csv   # Ranked replacement list
│   ├── employee_priority.csv      # Employee procurement ranking
│   ├── division_shortage.csv      # Division shortage analysis
│   └── procurement_recommendation.csv
├── notebooks/
│   ├── 01_Data_Preprocessing.ipynb
│   ├── 02_Model_Training.ipynb
│   ├── 03_Model_Inference.ipynb
│   └── 04_Dashboard_Analysis.ipynb
├── requirements.txt
└── README.md
```

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Jupyter
jupyter notebook

# 3. Execute notebooks in order:
#    01_Data_Preprocessing.ipynb  → Load, clean, engineer features
#    02_Model_Training.ipynb      → Train dual ML models
#    03_Model_Inference.ipynb     → Predict & rank assets
#    04_Dashboard_Analysis.ipynb  → KPIs, shortages, procurement
```

## ML Models

| Model | Type | Target | Purpose |
|-------|------|--------|---------|
| Replacement Model | RandomForestClassifier | Critical/High/Medium/Low | What to replace |
| Health Score Model | RandomForestRegressor | 0-100 continuous | How healthy is it |

## Key Features

- **Asset Health Score** - Weighted score (0-100) based on age, repairs, performance, depreciation, license, and remarks
- **Replacement Priority** - Multi-class classification with confidence probabilities
- **Employee Procurement Ranking** - Identifies employees without computers, ranked by employment status and work nature
- **Division Shortage Analysis** - Gap analysis between employee count and assigned computers
- **Budget Estimation** - Estimated procurement budget based on shortages and replacements
- **10+ Visualizations** - Feature importance, confusion matrix, distributions, and more

## Dependencies

pandas, numpy, matplotlib, scikit-learn, joblib
