import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


class MLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart System for Predictive Data Models")
        self.root.geometry("900x600")
        self.root.configure(bg="#1e1e2f")

        self.data = None

        # ===== TITLE =====
        title = tk.Label(
            root,
            text="Smart System for Predictive Data Models",
            font=("Segoe UI", 18, "bold"),
            bg="#1e1e2f",
            fg="#ffffff"
        )
        title.pack(pady=15)

        # ===== MAIN FRAME =====
        main_frame = tk.Frame(root, bg="#2a2a40")
        main_frame.pack(padx=20, pady=10, fill="both", expand=True)

        # ===== BUTTON FRAME =====
        btn_frame = tk.Frame(main_frame, bg="#2a2a40")
        btn_frame.pack(pady=15)

        tk.Button(
            btn_frame, text="📂 Load Dataset",
            width=20,
            font=("Segoe UI", 11, "bold"),
            bg="#4caf50",
            fg="white",
            relief="flat",
            command=self.load_data
        ).grid(row=0, column=0, padx=10)

        tk.Button(
            btn_frame, text="👀 Preview Dataset",
            width=20,
            font=("Segoe UI", 11, "bold"),
            bg="#2196f3",
            fg="white",
            relief="flat",
            command=self.preview_data
        ).grid(row=0, column=1, padx=10)

        tk.Button(
            btn_frame, text="🚀 Run Model Comparison",
            width=22,
            font=("Segoe UI", 11, "bold"),
            bg="#ff9800",
            fg="white",
            relief="flat",
            command=self.run_models
        ).grid(row=0, column=2, padx=10)

        # ===== OUTPUT BOX =====
        self.output = tk.Text(
            main_frame,
            height=20,
            width=100,
            bg="#121212",
            fg="#e0e0e0",
            font=("Consolas", 10),
            relief="flat"
        )
        self.output.pack(padx=10, pady=15)

    def load_data(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv")]
        )
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                messagebox.showinfo("Success", "Dataset loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def preview_data(self):
        if self.data is None:
            messagebox.showerror("Error", "Please load a dataset first")
            return

        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, "📊 DATASET PREVIEW (First 5 Rows)\n")
        self.output.insert(tk.END, "-" * 70 + "\n\n")
        self.output.insert(tk.END, str(self.data.head()))

    def run_models(self):
        if self.data is None:
            messagebox.showerror("Error", "Please load a dataset first")
            return

        try:
            # ===== FEATURES & TARGET =====
            X = self.data.iloc[:, :-1]
            y = self.data.iloc[:, -1]

            # ===== HANDLE CATEGORICAL DATA =====
            X = pd.get_dummies(X, drop_first=True)

            # ===== TRAIN TEST SPLIT =====
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # ===== FEATURE SCALING =====
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(alpha=1.0),
                "Lasso Regression": Lasso(alpha=0.1)
            }

            self.output.delete("1.0", tk.END)
            self.output.insert(tk.END, "📈 MODEL COMPARISON RESULTS\n")
            self.output.insert(tk.END, "=" * 70 + "\n\n")

            best_model = None
            best_score = -999

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                self.output.insert(
                    tk.END,
                    f"{name}\n"
                    f"➤ MSE : {mse:.2f}\n"
                    f"➤ R²  : {r2:.2f}\n\n"
                )

                if r2 > best_score:
                    best_score = r2
                    best_model = name

            self.output.insert(
                tk.END,
                f"🏆 BEST MODEL: {best_model} (Highest R² Score)\n"
            )

        except Exception as e:
            messagebox.showerror("Model Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = MLApp(root)
    root.mainloop()
