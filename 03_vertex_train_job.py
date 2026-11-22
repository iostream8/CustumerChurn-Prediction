"""Step 3 (CLOUD) - Run the SAME training script on Google Cloud Vertex AI.
Before running:  gcloud auth application-default login
                 gsutil cp data/customer_churn.csv gs://YOUR_BUCKET/data/customer_churn.csv"""
from google.cloud import aiplatform

PROJECT, REGION, BUCKET = "your-project-id", "us-central1", "gs://your-bucket-name"
aiplatform.init(project=PROJECT, location=REGION, staging_bucket=BUCKET)

job = aiplatform.CustomTrainingJob(
    display_name="churn-training",
    script_path="02_train_model.py",
    # Pick a prebuilt scikit-learn training image from the Vertex AI docs (use a recent one)
    container_uri="us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-0:latest",
    requirements=["pandas", "gcsfs", "matplotlib", "joblib", "google-cloud-storage"],
)
job.run(
    args=["--data", f"{BUCKET}/data/customer_churn.csv"],
    machine_type="n1-standard-4",
    replica_count=1,
    base_output_dir=f"{BUCKET}/churn-output",   # model.joblib lands in .../churn-output/model/
)
print("Done. Model saved in", f"{BUCKET}/churn-output/model/")
