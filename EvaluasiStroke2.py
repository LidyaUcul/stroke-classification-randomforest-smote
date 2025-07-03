import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# Fungsi untuk memuat dan memproses data
def load_and_preprocess_data(apply_smote=False):
    df = pd.read_csv("D:/Lidya Shafadhila/Machine Learning/healthcare-dataset-stroke-data.csv")

    df_clean = df.drop(columns=['id'])
    X = df_clean.drop(columns=['stroke'])
    y = df_clean['stroke']

    imputer = SimpleImputer(strategy='mean')
    X['bmi'] = imputer.fit_transform(X[['bmi']])

    categorical_cols = X.select_dtypes(include='object').columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    if apply_smote:
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)

    return X, y

# Fungsi pelatihan model
def train_models(use_smote=False):
    X, y = load_and_preprocess_data(apply_smote=use_smote)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced')
    }

    results = []
    matrices = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        })

        matrices[name] = confusion_matrix(y_test, y_pred)

    return pd.DataFrame(results), matrices

# Aplikasi Streamlit
def main():
    st.title("Stroke Prediction with SMOTE")

    use_smote = st.checkbox("Gunakan SMOTE untuk seimbangkan dataset")

    if st.button("Train Models"):
        with st.spinner("Melatih model..."):
            df_results, matrices = train_models(use_smote=use_smote)
        st.success("Model selesai dilatih!")
        st.subheader("Hasil Evaluasi Model")
        st.dataframe(df_results.style.format("{:.2%}", subset=["Accuracy", "Precision", "Recall", "F1 Score"]))

        st.subheader("Confusion Matrix")
        for model_name, matrix in matrices.items():
            st.markdown(f"**{model_name}**")
            fig, ax = plt.subplots()
            sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Stroke", "Stroke"], yticklabels=["No Stroke", "Stroke"])
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig)

if __name__ == "__main__":
    main()
