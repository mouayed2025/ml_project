from prefect import serve

from pipeline_prefect import all_flow, train_flow, evaluate_flow, code_flow

if __name__ == "__main__":
    # Planning cron (heure UTC) : "minute heure * * *" = une fois par jour
    all_deploy = all_flow.to_deployment(name="ml-pipeline-all", cron="0 8 * * *")
    train_deploy = train_flow.to_deployment(name="ml-pipeline-train", cron="0 9 * * *")
    evaluate_deploy = evaluate_flow.to_deployment(name="ml-pipeline-evaluate", cron="0 10 * * *")
    code_deploy = code_flow.to_deployment(name="ml-pipeline-code", cron="0 7 * * *")

    serve(all_deploy, train_deploy, evaluate_deploy, code_deploy)
