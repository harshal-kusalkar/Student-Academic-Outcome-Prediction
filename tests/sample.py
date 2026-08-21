import time
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

client = MlflowClient()

start = time.time()

model = client.get_model_version_by_alias(
    "student_dropout_model",
    "champion",
)

print("Metadata loaded in:", time.time() - start)
print("Version:", model.version)
print("Source:", model.source)
print("Run ID:", model.run_id)


model_uri = (
    "models:/student_dropout_model@champion"
)

start = time.time()

model = mlflow.pyfunc.load_model(
    model_uri
)

print(
    "Model loaded in:",
    time.time() - start,
)

print(
    "Model type:",
    type(model),
)

