import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

df = pd.read_csv("titanic.csv")

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
if df["Fare"].isnull().sum() > 0:
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

features = ["Pclass", "Age", "Fare", "SibSp", "Parch", "Sex_male", "Embarked_Q", "Embarked_S"]
X = df[features]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

log_accuracy = accuracy_score(y_test, log_pred)
log_precision = precision_score(y_test, log_pred)
log_recall = recall_score(y_test, log_pred)
log_f1 = f1_score(y_test, log_pred)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

root = tk.Tk()
root.title("🚢 Titanic Survival Dashboard")
root.geometry("900x600")

notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)

tab1 = ttk.Frame(notebook)
notebook.add(tab1, text="Dataset")

cols = list(df.columns)
tree = ttk.Treeview(tab1, columns=cols, show="headings", height=15)
for col in cols:
    tree.heading(col, text=col, anchor="center")
    if col == "Name":
        tree.column(col, anchor="center", width=200)
    else:
        tree.column(col, anchor="center", width=80)

for i in df.head(50).values.tolist():
    tree.insert("", "end", values=i)

tree.pack(fill="both", expand=True)

tab2 = ttk.Frame(notebook)
notebook.add(tab2, text="Exploratory Analysis")

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

df_plot = pd.read_csv("titanic.csv")
df_plot["Sex"].value_counts().plot(kind="bar", ax=axes[0], title="Passengers by Gender")
df_plot["Survived"].value_counts().plot(kind="bar", ax=axes[1], title="Survived vs Not")

fig.tight_layout()
canvas = FigureCanvasTkAgg(fig, master=tab2)
canvas.get_tk_widget().pack(fill="both", expand=True)
canvas.draw()

tab3 = ttk.Frame(notebook)
notebook.add(tab3, text="Model Performance")

columns = ("Model", "Accuracy", "Precision", "Recall", "F1-Score")
table = ttk.Treeview(tab3, columns=columns, show="headings", height=5)

for col in columns:
    table.heading(col, text=col, anchor="center")
    table.column(col, anchor="center", width=120)

table.insert("", "end", values=("Logistic Regression", f"{log_accuracy:.2f}", f"{log_precision:.2f}", f"{log_recall:.2f}", f"{log_f1:.2f}"))
table.insert("", "end", values=("Random Forest", f"{rf_accuracy:.2f}", f"{rf_precision:.2f}", f"{rf_recall:.2f}", f"{rf_f1:.2f}"))

table.pack(pady=20)

fig2, ax2 = plt.subplots(figsize=(5, 3))
models = ["Logistic Regression", "Random Forest"]
acc = [log_accuracy, rf_accuracy]
ax2.bar(models, acc, color=["skyblue", "orange"])
ax2.set_title("Accuracy Comparison")
canvas2 = FigureCanvasTkAgg(fig2, master=tab3)
canvas2.get_tk_widget().pack()
canvas2.draw()

tab4 = ttk.Frame(notebook)
notebook.add(tab4, text="Summary")

summary = scrolledtext.ScrolledText(tab4, wrap=tk.WORD, width=100, height=25, font=("Arial", 11))
summary.insert(tk.END, 
f"""
      Key Findings:

      Dataset contains Titanic passenger details including age, gender, class, and survival.
      Majority of passengers were male, but survival rate was higher for females.
      Passengers in higher classes (1st class) had better chances of survival.
      Logistic Regression Accuracy: {log_accuracy:.2f}
      Random Forest Accuracy: {rf_accuracy:.2f}
      Random Forest performed slightly better than Logistic Regression.
""")
summary.config(state="disabled")
summary.pack(fill="both", expand=True, padx=10, pady=10)

root.mainloop()
