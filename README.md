<p align="center">
  <img src="/pics/safe.png" alt="SEE PORT Logo" width="500"/>
</p>

# Fraud-detection-pipeline

## 1- 


---

## 2- Machine Learning & Streaming Pipeline Integration

The Machine Learning component in **TransactSafe** is designed as a distributed, real-time scoring engine integrated directly into **PySpark Structured Streaming**. The trained model is embedded inside Spark worker memory via **MLflow PySpark UDFs**, enabling high-throughput, parallel inference on live S3 data streams.


### 1. Data Topography & Feature Schema

The pipeline processes credit card transaction streams containing temporal, geospatial, financial, and cardholder identity signals:

<p align="center">
  <img src="pics/data_over.png" alt="Transaction Data Schema & Topography" width="800"/>
</p>


```text
• Timestamp (trans_date + trans_time) ──► Temporal context (hour of day, day of week, month)
• Cardholder & Geo (cc_num, lat, long)──► Who is transacting & customer location coordinates
• Spend Amount (amt)                  ──► Financial amount spent
• Category & Merchant (category, merch)──► Transaction category & merchant location (merch_lat, merch_long)
• Target Label (is_fraud)             ──► Ground-truth fraud flag (0 = Legit, 1 = Fraud)
```


### 2. How PySpark Handles Features & ML Scoring

During live streaming execution, PySpark worker nodes execute parallel feature transformations across micro-batches before evaluating model probabilities:


```text
                         [ Incoming S3 Stream ]
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                     PYSPARK DISTRIBUTED WORKER NODES                      │
│                                                                           │
│ 1. Vectorized Geospatial Features ──► Haversine Distance (Customer ↔ Merch)│
│ 2. Rolling Velocity Windows       ──► 1h / 24h Txn Counts & Spend Averages│
│ 3. Historical Target Encoding     ──► Broadcast Join (Category/Merchant Rate)│
│                                                                           │
│ 4. In-Memory MLflow UDF Scoring   ──► predict_udf(*20_features)          │
└───────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
          [ Probability >= 0.9980 Cutoff ──► Flagged Fraud Alert Outflow ]
```

### 3. Databricks Orchestration Workflows

The ML pipeline decouples 24/7 continuous streaming inference from periodic model retraining using two Databricks Job Workflows:

| Workflow Job | Script Location | Frequency / Trigger | Runtime Execution Mode | Core Responsibility |
| :--- | :--- | :--- | :--- | :--- |
| **Real-Time Detection** | [`realtime_detection.py`](./2-%20Machine%20Learning/src/scoring/realtime_detection.py) | **24/7 Continuous** Stream | PySpark Structured Streaming | Listens to S3 transaction streams, computes 20 features on the fly, scores rows via MLflow UDF, and outputs fraud alerts using streaming checkpoints. |
| **Weekly Retraining** | [`weekly_retrain.py`](./2-%20Machine%20Learning/src/retraining/weekly_retrain.py) | **Weekly Cron** (Scheduled) | PySpark Batch / Pandas Driver | Retrains CatBoost on historical Delta data with champion parameters (`scale_pos_weight=5.0`, `l2_leaf_reg=15`), evaluates test metrics, and promotes new model versions in MLflow. |

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                             DATABRICKS CLOUD                                │
│                                                                             │
│  [ 1: REAL-TIME DETECTION (24/7 Continuous Streaming) ]               │
│  S3 Raw Bucket ──► PySpark Structured Streaming ──► Feature Engineering     │
│                                                           │                 │
│  Alert Output ◄── Cutoff Threshold (>= 0.9980) ◄── MLflow Model UDF        │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  [ 2: WEEKLY MODEL RETRAINING (Scheduled Batch Workflow) ]             │
│  Historical Delta/S3 ──► Feature Pipeline ──► CatBoost Train ──► MLflow     │
│                                                                     │       │
│  MLflow Registry ("Production") ◄───────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
```

> 📖 **Deep ML Documentation & Experiments**  
> For in-depth ML specifics—including exploratory data analysis (EDA), hyperparameter tuning trials (SMOTENC vs. native weighting), CatBoost model benchmarks, and SHAP feature explainability—check the detailed ML subfolder README:  
>  
> ➡️ **[View Detailed Machine Learning Directory](./2-%20Machine%20Learning)**

---

## 3 -
