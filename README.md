# 3D Trajectory Analysis for Monkey Reaching Experiments

This repository contains code for analyzing 3D trajectory data from monkey reaching experiments and predicting movement onset and turn points using a bidirectional LSTM model.

## Note on Dataset

This repository contains **subset versions** of the feature files with 200 trials each (instead of the full 10,000-13,000+ trials). The subset files are provided to keep the repository size manageable for GitHub. The full processing pipeline and analysis code remains unchanged. The full dataset will be sent upon request.

## Project Overview

This research analyzes reaching movements from two monkeys performing tasks with 8 radial targets. The pipeline processes raw MATLAB data, extracts features, and trains neural networks to predict:
- **Movement onset time**: When the monkey begins moving after the go signal
- **Turn point time**: When the monkey changes direction in "update" trials (target changes mid-movement)

### Key Features
- 3D trajectory processing and alignment between subjects
- Bidirectional LSTM with Gaussian likelihood (predicts both mean and variance)
- Cross-subject generalization testing
- 10-fold cross-validation with different random seeds

## Pipeline Steps

### 1. MATLAB to DataFrame (`matlab_to_dataframe.ipynb`)

Converts raw `.mat` files to CSV format.

**Input**: `data_matlab/{nana,thina}/*.mat`

**Output**: `data_csv/FinalDfNana.csv`, `data_csv/FinalDfThina.csv`

Each trial contains:
- `time_milisecond`, `x`, `y`, `z`, `velocity`
- Reaction times: `RT` (no update), `RTu` (update trials)
- Movement times: `MT`, `MTu`
- Target IDs: `id_target`, `id_update`

**Run**:
```python
# Open the notebook and run all cells
jupyter notebook src/matlab_to_dataframe.ipynb
```

---

### 2. Coordinate System Alignment (`alignment.ipynb`)

Aligns Nana's and Thina's coordinate systems using the Orthogonal Procrustes algorithm.

**Input**: `data_csv/FinalDfNana.csv`, `data_csv/FinalDfThina.csv`

**Output**: `data_csv/FinalDfNana_aligned.csv`, `data_csv/FinalDfThina_aligned.csv`


**Run**:
```python
jupyter notebook src/alignment.ipynb
```

---

### 3. Feature Engineering (`compute_features.ipynb`)

Processes trajectories and computes features for the model.

**Input**: `data_csv/FinalDf{Nana,Thina}_aligned.csv`

**Output**: Pickled files in `features/` directory

**Available Files**:
- `Features_nana_subset` - 200 trials from Nana (13.0 MB)
- `Features_nana_aligned_subset` - 200 trials from Nana, aligned (18.9 MB)
- `Features_thina_subset` - 200 trials from Thina (13.3 MB)
- `Features_thina_aligned_subset` - 200 trials from Thina, aligned (14.3 MB)

**Note**: The full dataset files (595-1044 MB each) are excluded from this repository via `.gitignore`.

**Processing steps per trial**:
1. **Filter by criteria**:
   - Update trials: -320ms ≤ RTu ≤ 500ms
   - Non-update trials: -200ms ≤ RT ≤ 500ms
   - Movement duration ≤ 1200ms

2. **Window trajectories**:
   - Default: 200ms before go signal, 1500ms after
   - Normalize time within each trial (0 to 1)

3. **Smooth data**: Savitzky-Golay filter (window=5, poly=1)

4. **Recenter**: Shift trajectory so it starts at origin

5. **Rotate**: Align all trajectories so target 5 (rightward) is the reference direction

6. **Compute features**:
   - `rotation_x`, `rotation_y`, `rotation_z`: Rotated coordinates
   - `normalized_time`: Time normalized to [0, 1]
   - `velocity`: 3D velocity magnitude
   - `angle`: Angle between consecutive point triplets
   - `curvature`: 3D trajectory curvature
   - `distance_to_end`: Euclidean distance to endpoint

7. **Label events** as normalized time points:
   - `normalized_start_movement`: Movement onset
   - `normalized_stop_turn`: Turn point (update trials only)

**Output format** (pickled):
- `df_features`: DataFrame with trajectory features
- `start_move`: dict mapping trial ID → normalized start time
- `stop_turn`: dict mapping trial ID → normalized turn time (update trials)

**Run**:
```python
# Set monkey_name = 'nana' or 'thina' in the first cell
jupyter notebook src/compute_features.ipynb
```

---

### 4. Model Training

#### Model Architecture

```
Input: [batch, sequence_length, 7 features]
  ↓
Bidirectional LSTM (50 hidden units each direction)
  ↓
Adaptive Average Pooling → [batch, 100]
  ↓
Trajectory Type Embedding (5 types → 2 dims)
  ↓
Concatenate [LSTM features + embedding] → [batch, 102]
  ↓
FC Layer (102 → 100, ReLU)
  ↓
Split into two heads:
  - fc_mean → μ (predicted event time)
  - fc_logvar → log(σ²) (uncertainty estimate)
```

**Trajectory Types** (based on angular jump in update trials):
- Type 0: No update (straight reach)
- Type 45: 45° target jump
- Type 90: 90° target jump
- Type 135: 135° target jump
- Type 180: 180° target jump

**Loss Function**: Negative log-likelihood for Gaussian
```python
loss = exp(-logvar) * (target - mu)² + logvar
```

**Features used** (7 total):
```python
['rotation_x', 'rotation_y', 'rotation_z', 'normalized_time',
 'velocity', 'angle', 'distance_to_end']
```

---

#### Option A: Interactive Training (`model_train.ipynb`)

**Run**:
```python
jupyter notebook src/model_train.ipynb
```

---

#### Option B: Automated 10-Run Training (Python scripts)

**Movement Onset Model** (Thina train, Nana eval):
```bash
cd /path/to/FinalVersionPaper
python src/onset_model_train_eval.py
```

**Turn Point Model** (Nana train, Thina eval):
```bash
python src/turning_model_train_eval.py
```

**Turn Point Model** (Nana train/eval, no alignment):
```bash
python src/turn_10_times_no_align.py
```

**What happens**:
- Runs 10 training iterations with seeds 42, 43, ..., 51
- 80/20 train/test split per run
- 100 epochs, batch size 32, Adam optimizer (lr=0.001)
- Saves models: `stop_turn_model_run{1-10}.pth`
- Reports MAE (mean absolute error) in milliseconds

**Expected output**:
```
Run 1/10
Epoch 10/1000, Avg Loss: 0.1234
...
Run 1, MAE on Nana: 45.23 ms
Run 1, MAE on Thina: 52.67 ms
Model saved for run 1

...

Average MAE Nana: 47.15 ms
Average MAE Thina: 54.32 ms
```


## Coordinate System Conventions

1. **Original coordinates**: Raw sensor data (x, y, z in mm)
2. **Centered coordinates**: Origin at trajectory start point
3. **Rotated coordinates**: Aligned so target 5 (rightward) is reference
4. **Aligned coordinates** (cross-monkey): Kabsch-rotated to match reference monkey's target layout

---

## Scaling and Padding

- **Column-wise scaling**: Each of the 7 features is standardized (zero mean, unit variance) using training set statistics
- **Sequence padding**: Post-padding to max sequence length with zeros
- **Cross-dataset**: When evaluating on other monkey, use training set's scaling parameters

---

## MAE Calculation

Mean Absolute Error is calculated in **milliseconds** after denormalization:

```python
true_time_ms = normalized_value * (max_time[id] - min_time[id]) + min_time[id]
pred_time_ms = normalized_pred * (max_time[id] - min_time[id]) + min_time[id]
mae = |pred_time_ms - true_time_ms|
```



## Important Notes

- Update trials have both `start_move` and `stop_turn` labels
- Non-update trials only have `start_move` labels
- The `adjusted_time` column represents time relative to window start (not absolute)
- Feature files are pickled Python objects (use `pickle.load()` to read)
- Models are saved as PyTorch state dicts (`.pth` files)
