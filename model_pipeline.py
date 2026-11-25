import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def prepare_data(path="Churn_Modelling.csv"):
    """Charger et prétraiter les données."""
    df = pd.read_csv(path)
    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])
    df = df.drop(columns=["Surname", "Geography", "RowNumber", "CustomerId"])
    x = df.drop(columns=["Exited"])
    y = df["Exited"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=1
    )
    print("Données préparées :", x_train.shape, x_test.shape)
    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train):
    """Entraîner le modèle."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    print("Modèle entraîné")
    return model


def evaluate_model(model, x_test, y_test):
    """Évaluer les performances."""
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy :", accuracy * 100)
    print("Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred))
    return accuracy


def save_model(model, filename="classifier.joblib"):
    """Sauvegarder le modèle."""
    joblib.dump(model, filename)
    print("Modèle sauvegardé dans", filename)


def load_model(filename="classifier.joblib"):
    """Charger un modèle sauvegardé."""
    model = joblib.load(filename)
    print("Modèle chargé depuis", filename)
    return model


def predict_client(model, client):
    """Prédire si un client reste (0) ou part (1)."""
    colonnes = [
        "CreditScore",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    df_client = pd.DataFrame([client], columns=colonnes)
    prediction = model.predict(df_client)[0]
    if prediction == 1:
        print("Résultat : le client va PARTIR")
    else:
        print("Résultat : le client va RESTER")
    return prediction
