import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'patterncore'))

import tkinter as tk
from tkinter import messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
from scipy.stats import entropy

from pattern_ensemble import (
    train_ensemble,
    load_ensemble,
    predict_multilayer,
    append_outcome_to_csv,
    update_ensemble_on_new_outcome
)

# Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'processed_csvs', 'target_outcomes_master.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'data', 'models')

# Number classification
RED_NUMBERS = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}
BLACK_NUMBERS = {2,4,6,8,10,11,13,15,17,20,22,24,26,28,29,31,33,35}
GREEN_NUMBERS = {0}

# GUI Class
class PatternCoreGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PatternCore AI - Live Ensemble Predictor")

        self.models = load_ensemble()

        self.prediction_label = tk.Label(root, text="Prediction: --", font=("Helvetica", 24))
        self.prediction_label.pack(pady=10)

        self.confidence_label = tk.Label(root, text="Confidence: --", font=("Helvetica", 14))
        self.confidence_label.pack()

        self.entry = tk.Entry(root, font=("Helvetica", 18))
        self.entry.pack(pady=10)
        self.entry.bind("<Return>", lambda event: self.submit_outcome())

        self.submit_btn = tk.Button(root, text="Submit Outcome", command=self.submit_outcome)
        self.submit_btn.pack()

        self.stats_label = tk.Label(root, text="", font=("Helvetica", 12))
        self.stats_label.pack(pady=10)

        self.plot_frame = tk.Frame(root)
        self.plot_frame.pack()

        self.statbox_label = tk.Label(root, text="", font=("Consolas", 10), justify="left", anchor="w")
        self.statbox_label.pack(pady=10, fill="both")

        self.retrain_btn = tk.Button(root, text="Manual Retrain", command=self.manual_retrain)
        self.retrain_btn.pack(pady=5)

        self.refresh_prediction()

    def refresh_prediction(self):
        pred, conf, _ = predict_multilayer(self.models)
        self.prediction_label.config(text=f"Prediction: {pred}")
        self.confidence_label.config(text=f"Confidence: {conf:.2%}")
        self.update_stats()
        self.update_chart()
        self.update_prediction_stats()

    def submit_outcome(self):
        val = self.entry.get()
        if not val.isdigit():
            messagebox.showerror("Error", "Enter a valid number.")
            return
        val = int(val)
        if not (0 <= val <= 36):
            messagebox.showerror("Error", "Number must be between 0 and 36.")
            return

        append_outcome_to_csv(val)

        pred, conf, prob = predict_multilayer(self.models)
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prediction": pred,
            "actual": val,
            "confidence": round(conf, 4),
            "entropy": round(entropy(prob), 4)
        }
        log_path = os.path.join(MODEL_DIR, "prediction_log.csv")
        pd.DataFrame([row]).to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)

        self.models = update_ensemble_on_new_outcome(self.models)
        self.entry.delete(0, tk.END)
        self.refresh_prediction()

    def manual_retrain(self):
        self.models = train_ensemble()
        messagebox.showinfo("Retrain", "Manual retraining completed.")
        self.refresh_prediction()

    def update_stats(self):
        if not os.path.exists(CSV_PATH):
            return
        df = pd.read_csv(CSV_PATH)
        if df.empty:
            self.stats_label.config(text="No data yet.")
            return
        outcomes = df['outcome'].astype(int)
        total = len(outcomes)
        stats = {
            "Red %": outcomes.isin(RED_NUMBERS).sum() / total * 100,
            "Black %": outcomes.isin(BLACK_NUMBERS).sum() / total * 100,
            "Green %": outcomes.isin(GREEN_NUMBERS).sum() / total * 100,
            "Even %": (outcomes % 2 == 0).sum() / total * 100,
            "Odd %": (outcomes % 2 == 1).sum() / total * 100,
            "High %": (outcomes >= 19).sum() / total * 100,
            "Low %": (outcomes <= 18).sum() / total * 100,
            "Top Number": outcomes.value_counts().idxmax()
        }
        display = "\n".join([f"{k}: {v:.2f}%" if isinstance(v, float) else f"{k}: {v}" for k, v in stats.items()])
        self.stats_label.config(text=display)

    def update_chart(self):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        if not os.path.exists(CSV_PATH):
            return
        df = pd.read_csv(CSV_PATH)
        if df.empty:
            return
        counts = df['outcome'].value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(6, 2))
        ax.bar(counts.index, counts.values, color='skyblue')
        ax.set_title("Outcome Frequency")
        ax.set_xlabel("Number")
        ax.set_ylabel("Count")

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def update_prediction_stats(self):
        try:
            df = pd.read_csv(CSV_PATH)
            if 'outcome' not in df.columns or df.empty:
                self.statbox_label.config(text="No outcome data available.")
                return

            outcomes = pd.to_numeric(df['outcome'], errors='coerce').dropna().astype(int)
            total = len(outcomes)
            if total == 0:
                self.statbox_label.config(text="No valid outcomes.")
                return

            reds = outcomes.isin(RED_NUMBERS).sum()
            blacks = outcomes.isin(BLACK_NUMBERS).sum()
            greens = outcomes.isin(GREEN_NUMBERS).sum()
            evens = (outcomes % 2 == 0).sum()
            odds = (outcomes % 2 == 1).sum()
            highs = (outcomes >= 19).sum()
            lows = (outcomes <= 18).sum()
            most_common = outcomes.value_counts().idxmax()

            try:
                pred_log = pd.read_csv(os.path.join(MODEL_DIR, "prediction_log.csv"))
                pred_log = pred_log.tail(10)
                correct = (pred_log["prediction"] == pred_log["actual"]).sum()
                acc = correct / len(pred_log) * 100
                avg_conf = pred_log["confidence"].mean()
                last_entropy = pred_log["entropy"].iloc[-1]
                last_pred = pred_log["prediction"].iloc[-1]
                last_actual = pred_log["actual"].iloc[-1]
            except Exception:
                acc = avg_conf = last_entropy = 0
                last_pred = last_actual = "--"

            display_text = f"""
📊 Live Outcome Stats
--------------------------
Total Outcomes:        {total}
Most Frequent Number:  {most_common}

🎨 Color Distribution
Red:                   {reds / total * 100:.2f}%
Black:                 {blacks / total * 100:.2f}%
Green:                 {greens / total * 100:.2f}%

🔢 Number Properties
Even:                  {evens / total * 100:.2f}%
Odd:                   {odds / total * 100:.2f}%
High (19-36):          {highs / total * 100:.2f}%
Low (1-18):            {lows / total * 100:.2f}%

🤖 Prediction Stats
Recent Accuracy:       {acc:.2f}%
Average Confidence:    {avg_conf:.2f}
Last Entropy:          {last_entropy:.3f}
Last Prediction:       {last_pred} vs {last_actual}
"""
            self.statbox_label.config(text=display_text)

        except Exception as e:
            self.statbox_label.config(text=f"[ERROR] Could not load stats: {e}")

# === Run GUI ===
if __name__ == "__main__":
    root = tk.Tk()
    app = PatternCoreGUI(root)
    root.mainloop()
