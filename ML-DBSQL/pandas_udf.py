# Databricks notebook source
from pyspark.sql.functions import pandas_udf
from typing import Any, Union, Iterator, Tuple
import pandas as pd
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models import Model
from mlflow.utils import find_free_port
from mlflow.pyfunc.scoring_server.client import ScoringServerClient
from mlflow.models import get_flavor_backend
from mlflow.environment_variables import MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT

import subprocess
import collections
import threading
import sys

import numpy as np

import subprocess
import collections
import threading
import sys
import signal


# hard code the env options.
_ENV = 'conda'

# Return type is hardcoded to double
@pandas_udf("double")
def predict_simple(iterator: Iterator[Tuple[Union[pd.Series, pd.DataFrame], ...]]
    ) -> Iterator[pd.Series]:
  from mlflow.tracking.artifact_utils import _download_artifact_from_uri
  from pyspark.sql import Row
  import pandas as pd
  import mlflow
  import os
  import sys

  local_model_path = None
  model_metadata = None
  input_schema = None
  scoring_server_proc = None
  mlflow_client = None

  def set_up_mlflow_server_client(local_model_path):
    _MLFLOW_SERVER_OUTPUT_TAIL_LINES_TO_KEEP = 200

    pyfunc_backend = get_flavor_backend(
        local_model_path,
        env_manager='conda',
        install_mlflow=os.environ.get("MLFLOW_HOME") is not None,
        create_env_root_dir=True,
    )
    # launch scoring server
    server_port = find_free_port()
    scoring_server_proc = pyfunc_backend.serve(
        model_uri=local_model_path,
        port=server_port,
        host="127.0.0.1",
        timeout=MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT.get(),
        enable_mlserver=False,
        synchronous=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    server_tail_logs = collections.deque(maxlen=_MLFLOW_SERVER_OUTPUT_TAIL_LINES_TO_KEEP)

    def server_redirect_log_thread_func(child_stdout):
        for line in child_stdout:
            if isinstance(line, bytes):
                decoded = line.decode()
            else:
                decoded = line
            server_tail_logs.append(decoded)
            sys.stdout.write("[model server] " + decoded)

    server_redirect_log_thread = threading.Thread(
        target=server_redirect_log_thread_func,
        args=(scoring_server_proc.stdout,),
    )
    server_redirect_log_thread.setDaemon(True)
    server_redirect_log_thread.start()

    client = ScoringServerClient("127.0.0.1", server_port)

    try:
        client.wait_server_ready(timeout=90, scoring_server_proc=scoring_server_proc)
    except Exception:
        err_msg = "During spark UDF task execution, mlflow model server failed to launch. "
        if len(server_tail_logs) == _MLFLOW_SERVER_OUTPUT_TAIL_LINES_TO_KEEP:
            err_msg += (
                f"Last {_MLFLOW_SERVER_OUTPUT_TAIL_LINES_TO_KEEP} "
                "lines of MLflow model server output:\n"
            )
        else:
            err_msg += "MLflow model server output:\n"
        err_msg += "".join(server_tail_logs)
        raise MlflowException(err_msg)
    return client, scoring_server_proc

  # Args should be only one pd.DataFrame for prototyping.
  def _predict_row_batch(predict_fn, args):
    # Only support Struct(feature_fields) for prototyping.
    if len(args) != 1 and type(args[0]) != pd.DataFrame:
      raise Exception("Only single Struct column is supported for prototype")
    # Omitted the Array Type support in ML flow.
    pdf = args[0]
    result = predict_fn(pdf)
    if not isinstance(result, pd.DataFrame):
      result = pd.DataFrame(data=result)
    # Hard code to cast to float64
    result = result.select_dtypes(include=(np.number,)).astype(np.float64)
    # Omitted the Array Type support in ML flow.
    return result[result.columns[0]]


  for input_batch in iterator:
    # If the UDF is called with only multiple arguments,
    # the `input_batch` is a tuple which composes of several pd.Series/pd.DataFrame
    # objects.
    # If the UDF is called with only one argument,
    # the `input_batch` instance will be an instance of `pd.Series`/`pd.DataFrame`.
    if isinstance(input_batch, (pd.Series, pd.DataFrame)): 
        raise Exception("Need model input and uri")
    else:
        row_batch_args = input_batch

    # Download model once.
    if local_model_path is None: 
      if not isinstance(row_batch_args[0], pd.Series):
        raise Exception("Invalid uri.")
      model_uri = row_batch_args[0].iloc[0]
      local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)
      model_metadata = Model.load(os.path.join(local_model_path, MLMODEL_FILE_NAME))
      input_schema = model_metadata.get_input_schema()
    
    if _ENV == 'local':
      loaded_model = mlflow.pyfunc.load_model(local_model_path)
      def batch_predict_fn(pdf):
        return loaded_model.predict(pdf)
    else:
      if local_model_path is not None and mlflow_client is None:
        mlflow_client, scoring_server_proc = set_up_mlflow_server_client(local_model_path)
      if mlflow_client is None:
        raise Exception("Failed set up mlflow server.")
      def batch_predict_fn(pdf):
        return mlflow_client.invoke(pdf).get_predictions()
    
    # Exclude first column which is model uri.
    yield _predict_row_batch(batch_predict_fn, row_batch_args[1:])

spark.udf.register("predict_simple", predict_simple)


# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT predict_simple('models:/shitao-test/1', struct(date, item, sales)) as prediction, date, item, sales
# MAGIC FROM
# MAGIC -- hive_metastore.shitao_test.store_item_test 
# MAGIC hive_metastore.time_series.store_item_demand
