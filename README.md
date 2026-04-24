# 🍬 Nassau Candy — Factory Reallocation & Shipping Optimizer

A full data science + Streamlit dashboard project for intelligent factory
reassignment and shipping optimization.

---

## 📁 Project Files

```
nassau_project/
├── app.py                         ← Main Streamlit dashboard (5 modules)
├── data_engine.py                 ← Data loading & feature engineering
├── ml_engine.py                   ← ML models + simulation + recommendations
├── Nassau_Candy_Distributor.csv   ← Dataset (10,194 orders)
├── requirements.txt               ← Python dependencies
└── README.md                      ← This file
```

---

## ✅ Step-by-Step Setup Guide

### STEP 1 — Install Python (if not already installed)

Download Python 3.9 or higher from: https://www.python.org/downloads/

✅ Make sure to tick **"Add Python to PATH"** during installation on Windows.

To check your Python version, open a terminal and run:
```
python --version
```

---

### STEP 2 — Open a Terminal / Command Prompt

- **Windows:** Press `Win + R`, type `cmd`, press Enter
- **Mac:** Press `Cmd + Space`, type `Terminal`, press Enter
- **Linux:** Open your terminal app

---

### STEP 3 — Navigate to the Project Folder

After unzipping the project, navigate into it:

```bash
cd path/to/nassau_project
```

Example on Windows:
```bash
cd C:\Users\YourName\Downloads\nassau_project
```

Example on Mac/Linux:
```bash
cd ~/Downloads/nassau_project
```

---

### STEP 4 — (Recommended) Create a Virtual Environment

This keeps the project dependencies isolated from your system Python.

```bash
# Create virtual environment
python -m venv venv

# Activate it — Windows:
venv\Scripts\activate

# Activate it — Mac/Linux:
source venv/bin/activate
```

You will see `(venv)` appear at the start of your terminal line.

---

### STEP 5 — Install All Dependencies

```bash
pip install -r requirements.txt
```

This installs: `streamlit`, `plotly`, `scikit-learn`, `pandas`, `numpy`, `scipy`

It may take 1–3 minutes. You should see each package install successfully.

---

### STEP 6 — Run the Dashboard

```bash
streamlit run app.py
```

Your browser will open automatically at: **http://localhost:8501**

If it does not open automatically, copy that URL and paste it into your browser.

---

## 🎛️ How to Use the Dashboard

### Sidebar (left panel)
- **Global Filters:** Filter by Region, Ship Mode, Division
- **Optimizer Priority:** Slide between Speed (fast delivery) and Profit (high margin)
- **Top-N:** How many recommendations to generate

### Tab 1 — 📊 EDA & Insights
View all exploratory charts: lead time distributions, factory map, profit analysis,
ML model comparison scores.

### Tab 2 — 🏭 Factory Simulator
1. Select a **Product**, **Region**, and **Ship Mode**
2. Click **▶ Run Simulation**
3. See predicted lead times across ALL 5 factories with rankings

### Tab 3 — 🔮 What-If Analysis
1. Select a product and an **Alternative Factory** you want to test
2. Click **🔄 Compare Scenarios**
3. See a gauge chart, side-by-side bar chart, and a **Verdict**
   (Recommended / Trade-off / Not Recommended)

### Tab 4 — 🏆 Recommendations
1. Choose a Region and Ship Mode
2. Click **🏆 Generate Recommendations**
3. See ranked factory reassignment suggestions with confidence scores

### Tab 5 — ⚠️ Risk & Impact
- Profit margin risk bands by product
- Lead time heatmap (Region × Factory)
- High-risk route alerts
- Executive summary table

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'streamlit'` | Run `pip install -r requirements.txt` |
| `streamlit: command not found` | Try `python -m streamlit run app.py` |
| Browser does not open | Open http://localhost:8501 manually |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |
| Slow model training on first load | Wait ~30 seconds — it caches after that |

---

## 🤖 ML Models Used

| Model | Purpose |
|-------|---------|
| Linear Regression | Baseline — fast, interpretable |
| Random Forest | Handles non-linear patterns |
| Gradient Boosting | Best accuracy — auto-selected if highest R² |

The best model is automatically selected and shown in the sidebar.

---

## 🏭 Factories

| Factory | Location | Primary Products |
|---------|----------|-----------------|
| Lot's O' Nuts | Arizona | Wonka Bars (Nutty Crunch, Fudge Mallows, Scrumdiddlyumptious) |
| Wicked Choccy's | Georgia | Wonka Bars (Milk Choc, Triple Dazzle Caramel) |
| Sugar Shack | Minnesota | Laffy Taffy, SweeTARTS, Nerds, Fun Dip, Fizzy Lifting Drinks |
| Secret Factory | Illinois | Everlasting Gobstopper, Lickable Wallpaper, Wonka Gum |
| The Other Factory | Tennessee | Hair Toffee, Kazookles |

---

## 📄 Deliverables

- ✅ `app.py` — Full interactive Streamlit dashboard
- ✅ `data_engine.py` — Data pipeline (no external geo libraries needed)
- ✅ `ml_engine.py` — ML + simulation + optimization engine
- ✅ `Nassau_Candy_Research_Report.docx` — Research paper (separate file)
- ✅ `README.md` — This setup guide
