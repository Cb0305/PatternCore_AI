import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from scipy.stats import entropy

# === Paths ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'processed_csvs', 'target_outcomes_master.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'data', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# === Constants ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQUENCE_LEN = 120
NUM_CLASSES = 37
BATCH_SIZE = 32
EPOCHS = 10
ENSEMBLE_SIZE = 3

# === Model Definition ===
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=NUM_CLASSES, hidden_size=128, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# === Data Loader ===
def load_sequences(limit_last=None):
    df = pd.read_csv(CSV_PATH)
    if limit_last and len(df) > limit_last:
        df = df[-limit_last:]
    outcomes = df['outcome'].values
    one_hot = np.eye(NUM_CLASSES)[outcomes]
    X, y = [], []
    for i in range(len(one_hot) - SEQUENCE_LEN):
        X.append(one_hot[i:i + SEQUENCE_LEN])
        y.append(outcomes[i + SEQUENCE_LEN])
    return np.array(X), np.array(y)

# === Ensemble Training ===
def train_ensemble():
    X, y = load_sequences()
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X.reshape(-1, NUM_CLASSES)).reshape(-1, SEQUENCE_LEN, NUM_CLASSES)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    models = []
    for i in range(ENSEMBLE_SIZE):
        model = LSTMModel().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        model.train()

        for epoch in range(EPOCHS):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

        torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"pattern_model_{i}.pt"))
        models.append(model)
        print(f"✅ Model {i+1} trained & saved.")

    return models

# === Load Saved Ensemble ===
def load_ensemble():
    models = []
    for i in range(ENSEMBLE_SIZE):
        model = LSTMModel().to(DEVICE)
        path = os.path.join(MODEL_DIR, f"pattern_model_{i}.pt")
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        models.append(model)
    return models

# === Frequency Softmax (Statistical Layer) ===
def frequency_softmax(sequence, num_classes=NUM_CLASSES):
    recent = sequence[-150:] if len(sequence) > 150 else sequence
    counts = np.bincount(recent, minlength=num_classes)
    norm = counts / np.sum(counts) if np.sum(counts) > 0 else np.ones(num_classes) / num_classes
    exps = np.exp(norm - np.max(norm))
    return exps / np.sum(exps)

# === Multi-Layer DecisionCore Prediction ===
def predict_multilayer(models):
    df = pd.read_csv(CSV_PATH)
    outcomes = df['outcome'].values
    if len(outcomes) <= SEQUENCE_LEN:
        return None, 0.0, [0] * NUM_CLASSES

    X = np.eye(NUM_CLASSES)[outcomes]
    latest_seq = X[-SEQUENCE_LEN:]
    input_seq = latest_seq.reshape(1, SEQUENCE_LEN, NUM_CLASSES)

    scaler = MinMaxScaler()
    input_seq = scaler.fit_transform(input_seq.reshape(-1, NUM_CLASSES)).reshape(1, SEQUENCE_LEN, NUM_CLASSES)
    input_tensor = torch.tensor(input_seq, dtype=torch.float32).to(DEVICE)

    probs = []
    with torch.no_grad():
        for model in models:
            out = model(input_tensor)
            prob = torch.softmax(out, dim=1).cpu().numpy()[0]
            probs.append(prob)

    avg_lstm_prob = np.mean(probs, axis=0)
    lstm_prediction = int(np.argmax(avg_lstm_prob))
    lstm_conf = float(avg_lstm_prob[lstm_prediction])
    lstm_entropy = float(entropy(avg_lstm_prob))

    confidence_threshold = 0.55
    entropy_threshold = 3.0
    trust_lstm = lstm_conf >= confidence_threshold and lstm_entropy <= entropy_threshold

    freq_prob = frequency_softmax(outcomes)
    freq_prediction = int(np.argmax(freq_prob))

    if trust_lstm:
        final_blend = 0.8 * avg_lstm_prob + 0.2 * freq_prob
    else:
        final_blend = 0.4 * avg_lstm_prob + 0.6 * freq_prob

    final_prediction = int(np.argmax(final_blend))
    final_confidence = float(final_blend[final_prediction])

    print(f"\n[LSTM] Pred: {lstm_prediction}, Conf: {lstm_conf:.2f}, Entropy: {lstm_entropy:.3f}")
    print(f"[Freq] Pred: {freq_prediction}")
    print(f"[Blend] Pred: {final_prediction}, Final Conf: {final_confidence:.2f}")

    return final_prediction, final_confidence, final_blend

# === Update Each Model in Ensemble ===
def update_ensemble_on_new_outcome(models):
    df = pd.read_csv(CSV_PATH)
    if len(df) <= SEQUENCE_LEN:
        return models

    recent = df['outcome'].values[-(SEQUENCE_LEN + 1):]
    X_seq = np.eye(NUM_CLASSES)[recent[:-1]].reshape(1, SEQUENCE_LEN, NUM_CLASSES)
    y = np.array([recent[-1]])

    input_tensor = torch.tensor(X_seq, dtype=torch.float32).to(DEVICE)
    target_tensor = torch.tensor(y, dtype=torch.long).to(DEVICE)

    for i, model in enumerate(models):
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        criterion = nn.CrossEntropyLoss()

        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()

        print(f"🔁 Model {i+1} incrementally updated.")

    return models

# === Log Outcome ===
def append_outcome_to_csv(actual_outcome: int):
    color = "green" if actual_outcome == 0 else "red" if actual_outcome in {
        1, 3, 5, 7, 9, 12, 14, 16, 18,
        19, 21, 23, 25, 27, 30, 32, 34, 36
    } else "black"
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "outcome": actual_outcome,
        "color": color,
        "batch": "live"
    }
    df = pd.DataFrame([row])
    df.to_csv(CSV_PATH, mode='a', index=False, header=False)
    print(f"✅ Outcome {actual_outcome} logged.")

# === Main Live Loop ===
if __name__ == "__main__":
    models = train_ensemble()

    while True:
        pred, conf, _ = predict_multilayer(models)
        print(f"\n🔮 Prediction: {pred} | Confidence: {conf:.2%}")
        val = input("Enter actual outcome (0-36) or 'q' to quit: ")

        if val.strip().lower() == 'q':
            break
        if not val.isdigit() or not (0 <= int(val) <= 36):
            print("Invalid input. Try again.")
            continue

        actual = int(val)
        append_outcome_to_csv(actual)
        models = update_ensemble_on_new_outcome(models)
