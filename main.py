import argparse
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
    predict_client,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline ML Churn")
    parser.add_argument("--prepare", action="store_true", help="Préparer les données")
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle")
    parser.add_argument("--save", action="store_true", help="Sauvegarder le modèle")
    parser.add_argument(
        "--load", action="store_true", help="Charger le modèle sauvegardé"
    )
    parser.add_argument(
        "--predict", action="store_true", help="Prédire si un client reste ou part"
    )
    parser.add_argument(
        "--client",
        nargs=9,
        type=float,
        default=[850, 0, 43, 2, 125510.82, 1, 1, 1, 79084.10],
        help="Les 9 infos du client",
    )
    parser.add_argument("--all", action="store_true", help="Exécuter tout le pipeline")
    args = parser.parse_args()

    # Les données sont nécessaires pour toutes les étapes
    x_train, x_test, y_train, y_test = prepare_data()
    model = None

    if args.train or args.save or args.all:
        model = train_model(x_train, y_train)

    if args.save or args.all:
        save_model(model)

    if args.load or args.all:
        model = load_model()

    if args.evaluate or args.all:
        if model is None:
            model = load_model()
        evaluate_model(model, x_test, y_test)

    if args.predict:
        if model is None:
            model = load_model()
        # CreditScore, Gender(0=Female,1=Male), Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
        predict_client(model, args.client)
