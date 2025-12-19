import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle

# =========================
# Load data
# =========================
with open('features/Features_thina_aligned', 'rb') as f:
    df_features, start_move, stop_turn = pickle.load(f)

print(f"Number of features: {len(df_features['id'].unique())}")
df_features = df_features[(df_features['normalized_start_movement'].notna()) & 
                          (df_features['normalized_start_movement'] > 0.0005)]
start_move = {k: v for k, v in start_move.items() if v > 0.0005 and not pd.isna(v)}
dict_list = [('start_move', start_move)]

features = ['rotation_x', 'rotation_y','rotation_z','normalized_time','velocity','angle','distance_to_end']
X = df_features.drop(['type_trajectory'], axis=1)
grouped = X.groupby('id')
trajectories = [group[features].values for _, group in grouped]
ids = list(grouped.groups.keys())
trajectory_type = df_features.groupby('id')['type_trajectory'].first().values
unique_trajectory_types = np.unique(trajectory_type)
trajectory_type_to_index = {t: i for i, t in enumerate(unique_trajectory_types)}
trajectory_type_indices = np.array([trajectory_type_to_index[t] for t in trajectory_type])
num_trajectory_types = len(unique_trajectory_types)
max_time = df_features.groupby('id')['adjusted_time'].max()
min_time = df_features.groupby('id')['adjusted_time'].min()

# =========================
# Scaling & padding
# =========================
def scale_trajectories_columnwise(trajectories):
    all_traj = np.concatenate(trajectories, axis=0)
    means = all_traj.mean(axis=0)
    stds = all_traj.std(axis=0)
    stds[stds == 0] = 1.0
    scaled_trajectories = [(traj - means)/stds for traj in trajectories]
    return scaled_trajectories, means, stds

def pad_sequences(sequences, maxlen, dtype='float32', padding='post', value=0):
    padded_sequences = np.full((len(sequences), maxlen, sequences[0].shape[1]), value, dtype=dtype)
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen:
            padded_sequences[i] = seq[:maxlen]
        else:
            padded_sequences[i, :len(seq)] = seq
    return padded_sequences

X_scaled, means, stds = scale_trajectories_columnwise(trajectories)
max_sequence_length = max([len(seq) for seq in X_scaled])
X_padded = pad_sequences(X_scaled, max_sequence_length)

# =========================
# Dataset & model
# =========================
class TrajectoryDataset(Dataset):
    def __init__(self, sequences, types, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.types = torch.tensor(types, dtype=torch.int64)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.sequences[idx], self.types[idx], self.labels[idx]

class TrajectoryModel(nn.Module):
    def __init__(self, num_trajectory_types, number_of_features):
        super().__init__()
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
        embedded_type = self.flatten(self.embedding(type_input))
        combined = torch.cat((avg_pool_out, embedded_type), dim=1)
        x = torch.relu(self.fc1(combined))
        mu = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

def nll_criterion_gaussian(mu, logvar, target):
    return (torch.exp(-logvar)*(target - mu)**2 + logvar).mean()

# =========================
# Load Nana's dataset once
# =========================
with open('features/Features_nana_aligned', 'rb') as f:
    df_nana, start_move_nana, stop_turn_nana = pickle.load(f)

df_nana = df_nana[(df_nana['normalized_start_movement'].notna()) & 
                  (df_nana['normalized_start_movement'] > 0.0005)]
start_move_nana = {k: v for k, v in start_move_nana.items() if v > 0.0005 and not pd.isna(v)}

X_nana = df_nana.drop(['type_trajectory'], axis=1)
grouped_nana = X_nana.groupby('id')
trajectories_nana = [group[features].values for _, group in grouped_nana]
ids_nana = list(grouped_nana.groups.keys())
trajectory_type_nana = df_nana.groupby('id')['type_trajectory'].first().values
trajectory_type_indices_nana = np.array([trajectory_type_to_index.get(t,0) for t in trajectory_type_nana])
X_nana_scaled = [(traj - means)/stds for traj in trajectories_nana]
max_sequence_length_nana = max([len(seq) for seq in X_nana_scaled])
X_nana_padded = pad_sequences(X_nana_scaled, max_sequence_length_nana)

# =========================
# Device
# =========================
device = torch.device("cuda") if torch.cuda.is_available() else \
         torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", device)

# =========================
# Repeat 10 runs
# =========================
num_runs = 10
mae_results = []
mae_nana_results = []

for run in range(num_runs):
    print(f"\n========== Run {run+1}/{num_runs} ==========")
    X_train, X_test, id_train, id_test, type_train, type_test = train_test_split(
        X_padded, ids, trajectory_type_indices, test_size=0.2, random_state=42+run)

    for name, label_dict in dict_list:
        y_train = np.array([label_dict[_id] for _id in id_train])
        y_test = np.array([label_dict[_id] for _id in id_test])

        train_dataset = TrajectoryDataset(X_train, type_train, y_train)
        test_dataset = TrajectoryDataset(X_test, type_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = TrajectoryModel(num_trajectory_types, X_padded.shape[2]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        epochs = 100

        # Training
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for sequences, types, labels in train_loader:
                sequences, types, labels = sequences.to(device), types.to(device), labels.to(device)
                optimizer.zero_grad()
                mu, logvar = model(sequences, types)
                loss = nll_criterion_gaussian(mu.squeeze(), logvar.squeeze(), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if (epoch+1) % 10 == 0 or epoch==0:
                print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {running_loss/len(train_loader):.4f}")

        # Test set evaluation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for sequences, types, labels in test_loader:
                sequences, types, labels = sequences.to(device), types.to(device), labels.to(device)
                mu, logvar = model(sequences, types)
                all_preds.extend(mu.squeeze().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        total_mae = sum(abs(pred*(max_time[_id]-min_time[_id])+min_time[_id] -
                            lbl*(max_time[_id]-min_time[_id])-min_time[_id])
                        for pred,lbl,_id in zip(all_preds, all_labels, id_test))
        mean_mae = total_mae/len(id_test)
        print(f"Run {run+1} MAE (test set): {mean_mae:.4f}")
        mae_results.append(mean_mae)

        # =========================
        # Evaluate on Nana
        # =========================
        nana_dataset = TrajectoryDataset(X_nana_padded, trajectory_type_indices_nana,
                                        np.array([start_move_nana[_id] for _id in ids_nana]))
        nana_loader = DataLoader(nana_dataset, batch_size=32, shuffle=False)

        predictions_nana = []
        true_nana = []
        with torch.no_grad():
            for sequences, types, labels in nana_loader:
                sequences, types, labels = sequences.to(device), types.to(device), labels.to(device)
                mu, logvar = model(sequences, types)
                predictions_nana.extend(mu.squeeze().cpu().numpy())
                true_nana.extend(labels.cpu().numpy())
        total_mae_nana = sum(abs(pred*(df_nana.groupby('id')['adjusted_time'].max()[_id]-df_nana.groupby('id')['adjusted_time'].min()[_id])+
                                df_nana.groupby('id')['adjusted_time'].min()[_id] -
                                lbl*(df_nana.groupby('id')['adjusted_time'].max()[_id]-df_nana.groupby('id')['adjusted_time'].min()[_id])-
                                df_nana.groupby('id')['adjusted_time'].min()[_id])
                             for pred,lbl,_id in zip(predictions_nana,true_nana,ids_nana))
        mean_mae_nana = total_mae_nana / len(ids_nana)
        print(f"Run {run+1} MAE (Nana): {mean_mae_nana:.4f}")
        mae_nana_results.append(mean_mae_nana)

print("\n==== Final Results ====")
print("Test set MAE:", mae_results)
print("Mean Test MAE:", np.mean(mae_results))
print("Nana set MAE:", mae_nana_results)
print("Mean Nana MAE:", np.mean(mae_nana_results))
