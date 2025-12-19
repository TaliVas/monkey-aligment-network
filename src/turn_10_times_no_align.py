import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle

# Device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", device)

# Load Nana training data
with open('features/Features_nana', 'rb') as f:
    df_features, start_move, stop_turn = pickle.load(f)

print(f"Number of features before filtering: {len(df_features['id'].unique())}")
df_features = df_features[df_features['normalized_stop_turn'].notna()]
print(f"Number of features after filtering: {len(df_features['id'].unique())}")

stop_turn = {k: v for k, v in stop_turn.items() if v > 0.0005 and not pd.isna(v)}
dict_list = [('stop_turn', stop_turn)]

# Prepare annotated DataFrame
df_annotated = pd.DataFrame()
for feature, data in dict_list:
    df_temp = pd.DataFrame.from_dict(data, orient='index', columns=[feature]).reset_index()
    df_temp.columns = ['id', feature]
    if df_annotated.empty:
        df_annotated = df_temp
    else:
        df_annotated = df_annotated.merge(df_temp, on='id', how='outer')

# Features to use
features = ['rotation_x', 'rotation_y', 'rotation_z', 'normalized_time', 
            'velocity', 'angle', 'curvature', 'distance_to_end']

max_time = df_features.groupby('id')['adjusted_time'].max()
min_time = df_features.groupby('id')['adjusted_time'].min()

X = df_features.drop(['type_trajectory'], axis=1)
grouped = X.groupby('id')
trajectories = [group[features].values for _, group in grouped]
ids = list(grouped.groups.keys())

trajectory_type = df_features.groupby('id')['type_trajectory'].first().values
unique_trajectory_types = np.unique(trajectory_type)
trajectory_type_to_index = {t: i for i, t in enumerate(unique_trajectory_types)}
trajectory_type_indices = np.array([trajectory_type_to_index[t] for t in trajectory_type])
num_trajectory_types = len(unique_trajectory_types)

# Dataset class
class TrajectoryDataset(Dataset):
    def __init__(self, sequences, types, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32).to(device)
        self.types = torch.tensor(types, dtype=torch.int64).to(device)
        self.labels = torch.tensor(labels, dtype=torch.float32).to(device)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.types[idx], self.labels[idx]

# Model definition
class TrajectoryModel(nn.Module):
    def __init__(self, num_trajectory_types, number_of_features):
        super(TrajectoryModel, self).__init__()
        self.lstm = nn.LSTM(number_of_features, 50, bidirectional=True, batch_first=True)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.embedding = nn.Embedding(num_trajectory_types, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(100 + 2, 100)
        self.fc_mean = nn.Linear(100, 1)
        self.fc_logvar = nn.Linear(100, 1)
        
    def forward(self, x, type_input):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.permute(0, 2, 1)
        avg_pool_out = self.avg_pool(lstm_out).squeeze(-1)
        embedded_type = self.embedding(type_input)
        embedded_type = self.flatten(embedded_type)
        combined = torch.cat((avg_pool_out, embedded_type), dim=1)
        x = torch.relu(self.fc1(combined))
        mu = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# Loss
def nll_criterion_gaussian(mu, logvar, target, reduction='mean'):
    loss = torch.exp(-logvar) * (target - mu)**2 + logvar
    return loss.mean() if reduction=='mean' else loss.sum()

# Scaling and padding
def scale_trajectories_columnwise(trajectories):
    all_values_per_feature = [[] for _ in range(trajectories[0].shape[1])]
    for traj in trajectories:
        for i in range(traj.shape[1]):
            all_values_per_feature[i].extend(traj[:, i])
    means = np.array([np.mean(f) for f in all_values_per_feature])
    stds = np.array([np.std(f) if np.std(f)>0 else 1.0 for f in all_values_per_feature])
    scaled = [(traj - means)/stds for traj in trajectories]
    return scaled, means, stds

def scale_test_trajectories(test_trajectories, means, stds):
    return [(traj - means)/stds for traj in test_trajectories]

def pad_sequences(sequences, maxlen, value=0):
    padded = np.full((len(sequences), maxlen, sequences[0].shape[1]), value, dtype=np.float32)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq[:maxlen]
    return padded

# MAE calculation
def calculate_mae(outputs, labels, ids, max_time, min_time):
    total_mae = 0
    for pred, lbl, _id in zip(outputs, labels, ids):
        true_val = lbl * (max_time[_id]-min_time[_id]) + min_time[_id]
        pred_val = pred * (max_time[_id]-min_time[_id]) + min_time[_id]
        total_mae += abs(pred_val - true_val)
    return total_mae / len(ids)

# Lists to store MAEs
mae_nana_list = []
mae_thina_list = []
num_runs = 10

for run in range(num_runs):
    print(f"\n=== Run {run+1}/{num_runs} ===")
    
    # --- Train/Test split for Nana ---
    X_train, X_test, id_train, id_test, type_train_lst, type_test_lst = train_test_split(
        trajectories, ids, trajectory_type_indices, test_size=0.2, random_state=42+run)
    type_train = np.array(type_train_lst)
    type_test = np.array(type_test_lst)
    
    # Scale
    X_train_scaled, train_means, train_stds = scale_trajectories_columnwise(X_train)
    X_test_scaled = scale_test_trajectories(X_test, train_means, train_stds)
    
    # Pad
    max_len = max([len(seq) for seq in X_train_scaled])
    X_train_padded = pad_sequences(X_train_scaled, max_len)
    X_test_padded = pad_sequences(X_test_scaled, max_len)
    
    # Labels
    y_train = np.array([stop_turn[_id] for _id in id_train])
    y_test = np.array([stop_turn[_id] for _id in id_test])
    
    # Datasets and loaders
    train_dataset = TrajectoryDataset(X_train_padded, type_train, y_train)
    test_dataset = TrajectoryDataset(X_test_padded, type_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # --- Model ---
    model = TrajectoryModel(num_trajectory_types, X_train_padded.shape[2]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # --- Training ---
    epochs = 100
    for epoch in range(epochs):
        model.train()
        for sequences, types, labels in train_loader:
            sequences, types, labels = sequences.to(device), types.to(device), labels.to(device)
            optimizer.zero_grad()
            mu, logvar = model(sequences, types)
            loss = nll_criterion_gaussian(mu.squeeze(), logvar.squeeze(), labels)
            loss.backward()
            optimizer.step()
        if (epoch+1)%10==0 or epoch==0:
            print(f"Epoch {epoch+1}/{epochs} done")
    
    # --- Evaluate on Nana ---
    model.eval()
    preds_nana = []
    with torch.no_grad():
        for sequences, types, labels in test_loader:
            sequences, types, labels = sequences.to(device), types.to(device), labels.to(device)
            mu, _ = model(sequences, types)
            preds_nana.extend(mu.squeeze().tolist())
    
    mae_nana = calculate_mae(preds_nana, y_test, id_test, max_time, min_time)
    mae_nana_list.append(mae_nana)
    print(f"Run {run+1}, MAE on Nana: {mae_nana:.2f} ms")
    
    # --- Evaluate on Thina ---
    with open('features/Features_thina', 'rb') as f:
        df_thina, start_move_thina, stop_turn_thina = pickle.load(f)
    
    df_thina = df_thina[df_thina['normalized_stop_turn'].notna()]
    max_time_thina = df_thina.groupby('id')['adjusted_time'].max()
    min_time_thina = df_thina.groupby('id')['adjusted_time'].min()
    
    stop_turn_thina = {k:v for k,v in stop_turn_thina.items() if v>0.0005 and not pd.isna(v)}
    features_thina = ['rotation_x', 'rotation_y', 'rotation_z', 'normalized_time', 
                      'velocity', 'angle', 'curvature', 'distance_to_end']
    X_thina = df_thina.drop(['type_trajectory'], axis=1)
    grouped_thina = X_thina.groupby('id')
    trajectories_thina = [group[features_thina].values for _, group in grouped_thina]
    ids_thina = list(grouped_thina.groups.keys())
    
    # Keep only valid Thina IDs
    valid_indices = [i for i, traj_id in enumerate(ids_thina) if traj_id in stop_turn_thina]
    ids_thina = [ids_thina[i] for i in valid_indices]
    trajectories_thina = [trajectories_thina[i] for i in valid_indices]
    
    # Map trajectory types
    trajectory_type_thina = df_thina.groupby('id')['type_trajectory'].first().values
    trajectory_type_indices_thina = np.array([trajectory_type_to_index.get(t,0) for t in trajectory_type_thina])
    trajectory_type_indices_thina = np.array([trajectory_type_indices_thina[i] for i in valid_indices])
    
    # Scale and pad
    X_thina_scaled, _, _ = scale_trajectories_columnwise(trajectories_thina)
    max_len_thina = max([len(seq) for seq in X_thina_scaled])
    X_thina_padded = pad_sequences(X_thina_scaled, max_len_thina)
    
    # Labels
    y_thina = np.array([stop_turn_thina[_id] for _id in ids_thina])
    thina_dataset = TrajectoryDataset(X_thina_padded, trajectory_type_indices_thina, y_thina)
    thina_loader = DataLoader(thina_dataset, batch_size=32, shuffle=False)
    
    # Thina evaluation
    preds_thina = []
    with torch.no_grad():
        for sequences, types, labels in thina_loader:
            sequences, types, labels = sequences.to(device), types.to(device), labels.to(device)
            mu, _ = model(sequences, types)
            preds_thina.extend(mu.squeeze().tolist())
    
    mae_thina = calculate_mae(preds_thina, y_thina, ids_thina, max_time_thina, min_time_thina)
    mae_thina_list.append(mae_thina)
    print(f"Run {run+1}, MAE on Thina: {mae_thina:.2f} ms")
    
    # Save model
    torch.save(model.state_dict(), f"stop_turn_model_run{run+1}.pth")
    print(f"Model saved for run {run+1}")

# --- Summary ---
print("\nAll runs completed!")
print(f"Average MAE Nana: {np.mean(mae_nana_list):.2f} ms")
print(f"Average MAE Thina: {np.mean(mae_thina_list):.2f} ms")
