import argparse
import subprocess
import sys

from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from main import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
    predict_client,
)

# Fichiers du projet uniquement (évite de scanner venv/)
PROJECT_FILES = ["main.py", "model_pipeline.py", "pipeline_prefect.py"]

# Client exemple : CreditScore, Gender, Age, Tenure, Balance,
# NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
CLIENT_EXEMPLE = [850, 0, 43, 2, 125510.82, 1, 1, 1, 79084.10]


def run_command(command, fail_on_error=True):
    """Exécute une commande et affiche sa sortie dans les logs Prefect."""
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    if fail_on_error and result.returncode != 0:
        raise RuntimeError(f"Échec de la commande : {' '.join(command)}")
    return result.returncode


# ====================== TASKS : CODE ======================


@task(name="Récupération du projet")
def pull_project():
    run_command(["git", "pull", "origin", "main"])


@task(name="Installer les dépendances")
def install_dependencies():
    run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


@task(name="Formatage du code")
def format_code():
    run_command([sys.executable, "-m", "black", *PROJECT_FILES])


@task(name="Qualité du code")
def check_quality():
    run_command(
        [sys.executable, "-m", "flake8", "--max-line-length=120", *PROJECT_FILES],
        fail_on_error=False,
    )


@task(name="Sécurité du code")
def check_security():
    run_command(
        [sys.executable, "-m", "bandit", "-ll", *PROJECT_FILES],
        fail_on_error=False,
    )


@task(name="Tests unitaires")
def run_tests():
    run_command([sys.executable, "-m", "pytest", "tests", "-v"])


# ====================== TASKS : DATA / MODÈLE ======================


@task(name="Préparation des données", cache_policy=NO_CACHE)
def prepare_data_task():
    return prepare_data()


@task(name="Entraînement du modèle", cache_policy=NO_CACHE)
def train_model_task(x_train, y_train):
    return train_model(x_train, y_train)


@task(name="Sauvegarde du modèle", cache_policy=NO_CACHE)
def save_model_task(model):
    save_model(model)


@task(name="Chargement du modèle", cache_policy=NO_CACHE)
def load_model_task():
    return load_model()


@task(name="Évaluation du modèle", cache_policy=NO_CACHE)
def evaluate_model_task(model, x_test, y_test):
    return evaluate_model(model, x_test, y_test)


@task(name="Prédiction", cache_policy=NO_CACHE)
def predict_task(model, client):
    return predict_client(model, client)


# ====================== FLOWS ======================


@flow(name="code", log_prints=True)
def code_flow():
    format_code()
    check_quality()
    check_security()
    run_tests()


@flow(name="train", log_prints=True)
def train_flow():
    x_train, x_test, y_train, y_test = prepare_data_task()
    model = train_model_task(x_train, y_train)
    return model


@flow(name="evaluate", log_prints=True)
def evaluate_flow():
    _, x_test, _, y_test = prepare_data_task()
    model = load_model_task()
    evaluate_model_task(model, x_test, y_test)


@flow(name="predict", log_prints=True)
def predict_flow():
    model = load_model_task()
    predict_task(model, CLIENT_EXEMPLE)


@flow(name="all", log_prints=True)
def all_flow():
    pull_project()
    install_dependencies()
    format_code()
    check_quality()
    check_security()
    run_tests()
    x_train, x_test, y_train, y_test = prepare_data_task()
    model = train_model_task(x_train, y_train)
    save_model_task(model)
    evaluate_model_task(model, x_test, y_test)
    predict_task(model, CLIENT_EXEMPLE)


# ====================== LANCEMENT ======================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Prefect - Churn")
    parser.add_argument(
        "--flow",
        required=True,
        choices=["all", "train", "entrainement", "evaluate", "code", "predict"],
        help="Flow à exécuter",
    )
    args = parser.parse_args()

    if args.flow == "all":
        all_flow()
    elif args.flow in ("train", "entrainement"):
        train_flow()
    elif args.flow == "evaluate":
        evaluate_flow()
    elif args.flow == "code":
        code_flow()
    elif args.flow == "predict":
        predict_flow()
