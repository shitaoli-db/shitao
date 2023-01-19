# Databricks notebook source
import mlflow

loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri='runs:/06c4bd1f805e47b49476ca9d468503e3/model', result_type='double')
spark.udf.register("mlflow_predict", loaded_model)

# COMMAND ----------

# MAGIC %python
# MAGIC 
# MAGIC from pyspark.sql.functions import udf
# MAGIC from pyspark.sql.types import DoubleType
# MAGIC from mlflow.models.model import MLMODEL_FILE_NAME
# MAGIC from mlflow.models import Model
# MAGIC from mlflow.utils import find_free_port
# MAGIC 
# MAGIC import numpy as np
# MAGIC 
# MAGIC import subprocess
# MAGIC import collections
# MAGIC import threading
# MAGIC import sys
# MAGIC 
# MAGIC 
# MAGIC class _EnvManager:
# MAGIC   LOCAL = "local"
# MAGIC   CONDA = "conda"
# MAGIC   VIRTUALENV = "virtualenv"
# MAGIC 
# MAGIC 
# MAGIC # UDF does not take kwargs on the calling side.
# MAGIC def predict_simple(model_uri, x, env_manager='local'):
# MAGIC   from mlflow.tracking.artifact_utils import _download_artifact_from_uri
# MAGIC   from pyspark.sql import Row
# MAGIC   import pandas as pd
# MAGIC   import mlflow
# MAGIC   import os
# MAGIC   import sys
# MAGIC   
# MAGIC   local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)
# MAGIC   model_metadata = Model.load(os.path.join(local_model_path, MLMODEL_FILE_NAME))
# MAGIC   input_schema = model_metadata.get_input_schema()
# MAGIC 
# MAGIC   if env_manager != _EnvManager.LOCAL:
# MAGIC     from mlflow.pyfunc.scoring_server.client import ScoringServerClient
# MAGIC     from mlflow.models import get_flavor_backend
# MAGIC     from mlflow.environment_variables import MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT
# MAGIC     
# MAGIC     import subprocess
# MAGIC     import collections
# MAGIC     import threading
# MAGIC     import sys
# MAGIC     _MLFLOW_SERVER_OUTPUT_TAIL_LINES_TO_KEEP = 200
# MAGIC 
# MAGIC     pyfunc_backend = get_flavor_backend(
# MAGIC         local_model_path,
# MAGIC         env_manager='conda',
# MAGIC         install_mlflow=os.environ.get("MLFLOW_HOME") is not None,
# MAGIC         create_env_root_dir=True,
# MAGIC     )
# MAGIC     # launch scoring server
# MAGIC     server_port = find_free_port()
# MAGIC     scoring_server_proc = pyfunc_backend.serve(
# MAGIC         model_uri=local_model_path,
# MAGIC         port=server_port,
# MAGIC         host="127.0.0.1",
# MAGIC         timeout=MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT.get(),
# MAGIC         enable_mlserver=False,
# MAGIC         synchronous=False,
# MAGIC         stdout=subprocess.PIPE,
# MAGIC         stderr=subprocess.STDOUT,
# MAGIC     )
# MAGIC 
# MAGIC     server_tail_logs = collections.deque(maxlen=_MLFLOW_SERVER_OUTPUT_TAIL_LINES_TO_KEEP)
# MAGIC 
# MAGIC     def server_redirect_log_thread_func(child_stdout):
# MAGIC         for line in child_stdout:
# MAGIC             if isinstance(line, bytes):
# MAGIC                 decoded = line.decode()
# MAGIC             else:
# MAGIC                 decoded = line
# MAGIC             server_tail_logs.append(decoded)
# MAGIC             sys.stdout.write("[model server] " + decoded)
# MAGIC 
# MAGIC     server_redirect_log_thread = threading.Thread(
# MAGIC         target=server_redirect_log_thread_func,
# MAGIC         args=(scoring_server_proc.stdout,),
# MAGIC     )
# MAGIC     server_redirect_log_thread.setDaemon(True)
# MAGIC     server_redirect_log_thread.start()
# MAGIC 
# MAGIC     client = ScoringServerClient("127.0.0.1", server_port)
# MAGIC 
# MAGIC     try:
# MAGIC         client.wait_server_ready(timeout=90, scoring_server_proc=scoring_server_proc)
# MAGIC     except Exception:
# MAGIC         err_msg = "During spark UDF task execution, mlflow model server failed to launch. "
# MAGIC         if len(server_tail_logs) == _MLFLOW_SERVER_OUTPUT_TAIL_LINES_TO_KEEP:
# MAGIC             err_msg += (
# MAGIC                 f"Last {_MLFLOW_SERVER_OUTPUT_TAIL_LINES_TO_KEEP} "
# MAGIC                 "lines of MLflow model server output:\n"
# MAGIC             )
# MAGIC         else:
# MAGIC             err_msg += "MLflow model server output:\n"
# MAGIC         err_msg += "".join(server_tail_logs)
# MAGIC         raise MlflowException(err_msg)
# MAGIC 
# MAGIC     def batch_predict_fn(pdf):
# MAGIC         return client.invoke(pdf).get_predictions()
# MAGIC   else:
# MAGIC     sys.stdout.write("[local model] Downloading model.\n")
# MAGIC     loaded_model = mlflow.pyfunc.load_model(local_model_path)
# MAGIC     def batch_predict_fn(pdf):
# MAGIC       return loaded_model.predict(pdf)
# MAGIC 
# MAGIC 
# MAGIC   if input_schema is None:
# MAGIC     pass
# MAGIC   else:
# MAGIC     names = input_schema.input_names()
# MAGIC   pdf = None
# MAGIC   # SQL Struct (Row) -> Pandas
# MAGIC   if type(x) is Row:
# MAGIC     pdf = pd.DataFrame(
# MAGIC       data={
# MAGIC         name:[x[name] if name in x else None] for name in names
# MAGIC       },
# MAGIC       columns=names
# MAGIC     )
# MAGIC     # THIS iS ANOYING BUG, PANDAS INFER PYTHON INT TO INT64 WHILE MLFLOW FORCE INT TO INT 32
# MAGIC     # Should use more grace way to cast
# MAGIC     input_types = input_schema.pandas_types()
# MAGIC     for i, name in enumerate(names):
# MAGIC       if input_types[i] == np.dtype('int32'):
# MAGIC         pdf[name] = pdf[name].astype(input_types[i])
# MAGIC   else:
# MAGIC     # Implementation of *args -> Pandas DF should be similar, ommited here.
# MAGIC     pass
# MAGIC   sys.stdout.write("[local model] Start model inference.\n")
# MAGIC   if pdf is not None:
# MAGIC     result = batch_predict_fn(pdf)
# MAGIC     sys.stdout.write("[local model] Finished model inference.\n")
# MAGIC     # Hardcode to get the result, spark python udf requires pickle.
# MAGIC     # Ideally should handle using utils like to_arrow_type etc.
# MAGIC     return float(result[0])
# MAGIC 
# MAGIC spark.udf.register("predict_simple", predict_simple)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT predict_simple('runs:/cb65b9c9ed7c4f2da73464aabac7e40e/model', struct(date, item, sales), 'conda') FROM hive_metastore.shitao_test.store_item_test
