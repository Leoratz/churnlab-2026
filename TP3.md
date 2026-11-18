# TP03 — Track, Compare and Recover ML Experiments with MLflow

ESILV — Machine Learning Operations  
Duration 90 minutes  
Dataset IBM Telco Customer Churn  
Starting point your own TP02 project and working scikit-learn Pipeline  
Format independent implementation, with instructor support

 Your team has trained several churn models. A colleague asks “Which model did you choose What produced it Can I load it without running training again” Provide evidence.

## Your mission

Extend your existing training workflow with MLflow. By the end, you should be able to find a training run, inspect its configuration and validation results, recover its complete fitted pipeline, and hand it over through a registered model alias.

You receive requirements and short code hints, not a completed implementation. Adapt the hints to your own modules and variable names. A placeholder such as `...` means work you must implement.

Keep the scope manageable reuse your TP02 models, preprocessing and evaluation functions. No new algorithms or large hyperparameter searches are needed.

### Learning outcomes

You should be able to

- Explain Experiment, Run, Parameter, Metric, Artifact and Model.
- Add explicit tracking to an existing training program.
- Compare runs under the same evaluation protocol.
- Connect a model to its code and data identities.
- Distinguish database metadata, artifact files and dataset records.
- Register the selected model, assign an alias and load it in a fresh process without refitting.

### Before starting

Your TP02 project must already train at least one complete preprocessing-and-classifier pipeline from a terminal. If model comparison is unfinished, use one working classifier with three modest configurations for this TP.

Continue with the same six features `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService` and `PaymentMethod`. Keep `Churn = Yes` mapped to class `1`; exclude the target and `customerID` from features.

Retain TP02's trainvalidationtest customer allocation. Fit preprocessing on training only. Compare candidates on validation, with average precision as the primary metric, computed from continuous positive-class scores. Also report precision, recall and F1 at the same documented threshold. Keep the test set reserved.

If TP02 documented a different primary metric, confirm the choice with the instructor and apply it consistently; do not switch metrics after seeing which model wins.

## Suggested timing

 Time  Milestone 
------
 0–15 min  Start the server, create a run and inspect storage 
 15–40 min  Record parameters, metrics, artifacts and dataset inputs 
 40–55 min  Compare at least three existing candidate configurations 
 55–80 min  Log, register and reload the selected pipeline through an alias 
 80–90 min  Restart, verify persistence and collect handover evidence 

## 1. Start tracking locally — 15 minutes

Add MLflow to your existing environment, from the project root

```bash
uv add mlflow=3,4
uv run mlflow --version
```

Use MLflow 3.x for the hints below and retain the resolved version in your lockfile. Preserve the dependencies your TP02 code needs.

In a first terminal, from the project root

```bash
uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlitemlflow.db --artifacts-destination .mlartifacts
```

Keep this terminal running and open [http127.0.0.15000](http127.0.0.15000). Use a second terminal for your training commands.

The server provides the tracking API and browser UI. SQLite stores run metadata; the artifact directory stores files such as recorded models. These are local to this project and machine.

### Observe the two storage locations

 Component  What it holds in this exercise 
------
 Backend store `mlflow.db`  Experimentrun records, parameters, metrics, tags and registry metadata 
 Artifact store `mlartifacts`  Model files, confusion matrices and other logged files 
 Tracking server  Receives loggingquery requests and serves the UI 
 Training client  Executes your training code and sends records to the server 

Locate the database and artifact directory after your first runmodel is logged. Do not modify database rows or generated artifact files manually. Keep this table for the restart experiment in section 5.

Question if a colleague receives only `mlflow.db`, what else would they need to load the recorded pipeline

### Code hint — connect the training client

Add this setup before creating runs

```python
import mlflow

mlflow.set_tracking_uri(http127.0.0.15000)
mlflow.set_experiment(telco-churn-tp03)
```

Create a small first run with `mlflow.start_run()` and log one parameter. Find it in the UI before changing the full training workflow.

Checkpoint you can point to your experiment and the ID of one run.

Explain what is the difference between an experiment and a run Is the browser page executing your training code

## 2. Record what produced each result — 25 minutes

Wrap one candidate's training and validation evaluation in one run. If your script loops over candidates, create a separate run for each candidate.

Use explicit logging for this core exercise so that you can explain what is recorded.

 Category  Required information 
------
 Parameters  Estimator type, important hyperparameters, split seed, decision threshold 
 Metrics  `val_average_precision`, `val_precision`, `val_recall`, `val_f1` 
 Tags  Git commit and SHA-256 of the original CSV 
 Artifacts  Validation confusion matrix and a compact configurationprovenance JSON 
 Dataset inputs  Training and validation dataset metadata with distinct contexts 
 Selected model, in section 4  Complete fitted preprocessing-and-classifier pipeline 

Reuse TP01TP02's hash and provenance logic. The JSON should also identify feature names, class mapping, splitcustomer-allocation reference and evaluation rules. A file hash identifies bytes; it does not store or recover the CSV.

### Code hint — insert tracking around your own workflow

```python
with mlflow.start_run(run_name=run_name) as run
    mlflow.log_params({
        model_type ...,
        random_state ...,
        threshold ...,
        # TODO add the actual hyperparameters used.
    })
    mlflow.set_tags({
        git_commit ...,
        data_sha256 ...,
    })

    # TODO fit your existing complete pipeline on training data.
    # TODO evaluate validation data with your TP02 evaluation function.
    mlflow.log_metrics(validation_metrics)
    mlflow.log_artifact(str(confusion_matrix_path))
    mlflow.log_dict(provenance, provenance.json)
    print(Run ID, run.info.run_id)
```

The snippet is incomplete. `validation_metrics` must contain numeric values and consistently named keys. Pass only actual candidate settings, not the settings you intended to use.

Find the score column for class `1` from the estimator's `classes_`; do not assume a column position without checking. Average precision uses continuous scores. Precision, recall and F1 use your fixed threshold. State the confusion matrix's class ordering.

### Code hint — associate data with the run

Create dataset records from your existing split DataFrames; do not change the split to add tracking

```python
train_dataset = mlflow.data.from_pandas(
    train_df,
    name=telco_train,
    targets=Churn,
)

# Inside the active candidate run
mlflow.log_input(train_dataset, context=training)
# TODO create and log a validation dataset with context=validation.
```

Here `train_df` contains the target for describing the dataset. The estimator still receives only the six agreed features, with the separately encoded target from TP02. The raw target column is not a feature.

Inspect the dataset name, digest and schema in the UI where available. These describe the dataset used; logging them does not automatically preserve the full DataFrame or implement dataset version control. Keep the original CSV SHA-256 and exact splitcustomer-allocation records. MLflow's dataset digest is not interchangeable with your original-file SHA-256.

Question what evidence distinguishes two different trainvalidation allocations made from the same original CSV

Checkpoint inspect one run in the UI and locate its parameters, metrics, tags, files and trainingvalidation dataset records. Explain the difference between a metric and an artifact.

Think about it

- Does recording a random seed guarantee identical results on every machine
- Can someone recover the training data from its hash alone
- What happens to traceability if you log a commit while training uncommitted edits

For this exercise, commit your training changes before the final comparison runs. If there are relevant uncommitted changes, record that fact; a commit ID alone does not describe them.

## 3. Compare at least three candidate runs — 15 minutes

Produce at least three completed candidate runs, using either different TP02 classifiers or different configurations of a working classifier. Keep the six features, customer allocation, evaluation definitions and threshold policy fixed.

Use descriptive names such as `logistic-C1` or `forest-depth5`; the exact names depend on what you actually run. You may add a baseline run if it is already available.

Select the runs in the UI and compare them. Record your decision in a small table

 Run ID  Candidate  changed setting  Validation AP  Validation recall  Validation F1  Decision 
------------------
 Complete with your results      

Choose using your predefined primary metric. If candidates are close, explain whether simplicity, fit duration or operational constraints influence your choice. This is a classroom comparison on one split, not proof of a statistically significant improvement.

Checkpoint save a screenshot of the comparison and write 3–5 sentences explaining the selection.

Explain why would results from different validation populations be harder to compare Why should you avoid selecting repeatedly on the test set

## 4. Log, register and recover the complete model — 25 minutes

Record the fitted preprocessing plus classifier together using `mlflow.sklearn.log_model`. Saving only a `joblib` file as a generic artifact does not demonstrate MLflow's model logging API.

Ideally log the pipeline inside every candidate's run. For the core requirement, at least the selected pipeline must be logged in its corresponding run. You may resume that run to attach the existing fitted pipeline, or repeat the exact selected configuration as a clearly identified confirmation run. Do not attach a different model to an earlier run.

### Code hint — log a fitted pipeline

Inside the appropriate active run, after fitting

```python
import mlflow.sklearn

model_info = mlflow.sklearn.log_model(
    sk_model=fitted_pipeline,
    name=churn_pipeline,
    input_example=X_train.head(3),
    serialization_format=cloudpickle,
)
print(Model URI, model_info.model_uri)
```

Use original, unencoded feature columns in the documented input types for the example, without `Churn` or `customerID`. MLflow can infer a signature from this example. Inspect the stored input schema and explain what it describes.

Record both the run ID and the returned model URI. They identify different objects. In MLflow 3.x, a logged model may appear in the Models area associated with its source run.

If your pipeline uses custom transformers, the loading environment must be able to import their modules. Retain your project package and compatible dependencies.

### Register the selected model for handover

Register your selected logged pipeline under the name `telco-churn`. Assign the alias `candidate` to the selected registered version. Using the Registry UI is sufficient; API automation is an optional extension.

Use the logged model's page or its source run to find the registration entry point; exact labels vary across MLflow 3.x versions. Inspect the registered model's version list, then add the alias to the intended version.

If you prefer the API, investigate

```python
# After logging the selected pipeline, with the same tracking server
version = mlflow.register_model(model_info.model_uri, telco-churn)
client = mlflow.MlflowClient()
# TODO use client.set_registered_model_alias(...)
#       to point candidate at version.version.
```

Record the actual returned version; do not assume it is always version 1.

 Identifier  Question it answers 
------
 Run ID  Which execution produced the result 
 Logged model URI  Which recorded model artifact should be loaded 
 Registered name + version  Which version belongs to the managed model `telco-churn` 
 Alias `candidate`  Which version is currently selected for this named role 

An alias is a mutable reference, not a new copy of the model and not an automatic deployment. Keep the specific version alongside the alias for reproducible handover.

### Code hint — load through the alias in a fresh process

Create an inference-only entry point using your own project structure

```python
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri(http127.0.0.15000)
model_uri = modelstelco-churn@candidate
loaded_pipeline = mlflow.sklearn.load_model(model_uri)
new_customers = ...  # Buildload a small batch with the agreed input schema.
predictions = loaded_pipeline.predict(new_customers)
```

Run this in a new terminal process, not merely another cell in the training notebook. Keep the tracking server running. Your process must not call `fit` or import a module that starts training on import.

Apply the same documented stateless input parsing as TP02. “Unencoded inputs” does not mean malformed CSV values may bypass your input-type rules.

Use the same few records to compare original and reloaded predictions. Class predictions should agree; use a stated tolerance if comparing probabilities. Load only your own trusted model artifact.

Checkpoint show the source run ID, logged model URI, registered version, alias, inference command and prediction agreement. Explain why successful loading in a new process is stronger evidence than predicting again in the training notebook.

Question if the alias later points to a new version, does an already-loaded Python object automatically change Explain when another load would be needed.

## 5. Verify persistence, submit and explain — 10 minutes

Stop the tracking server and restart it from the same project root using the same command. Confirm that your runs, registered model version and alias remain available, then run your inference entry point again.

Do not delete or rename storage to simulate failures. Instead, answer what would become unavailable if the database survived but the model files were lost What would be missing if only the model files survived

Checkpoint record a short beforeafter restart observation; an extra screenshot is not required.


Update your README with the exact commands for server startup, tracked training and inference. State where local tracking data are stored and how to recover your selected model.

Keep the virtual environment and generated tracking storage out of Git. For the layout above, review these ignore patterns

```gitignore
.venv
mlflow.db
mlflow.db-
mlartifacts
mlruns
```

Retain the local database and artifact files for the demonstration; ignoring them in Git does not mean deleting them. Stopping the server does not delete the stored runs. Restart it from the same directory with the same options.

A run ID or local model URI alone is insufficient for a classmate on another machine. Document this limitation; sharing a team server or exporting the required artifacts and environment is additional work.

### Deliverables

Submit your repository link or source archive, plus a short `reportstp03_summary.md` containing

1. A screenshot comparing at least three candidate runs and the selection table.
2. A selected-run screenshot (or two cropped views) showing parameters, validation metrics and trainingvalidation dataset records.
3. A Registry screenshot showing `telco-churn`, its selected version and `candidate` alias.
4. The source run ID, logged model URI, registered version, alias, Git commit and original data hash.
5. The validation-based selection reason and the exact fresh-process inference command, with evidence of prediction agreement.
6. A brief restart observation and answers to “What do the database and artifact store each preserve What must a colleague on another machine obtain separately”

Submit screenshots and small evidence files through the course platform. Do not assume the instructor can access your machine's localhost address. Be ready to demonstrate the selected run and its recorded model.

### Assessment

 Criterion  Points 
------
 Local setup, Experiment  Run organization and storagerestart explanation  1 
 Useful parameters, metrics, tags, artifacts and dataset inputs  3 
 Three fair candidate runs and justified validation-based choice  2 
 Complete logged pipeline, registered versionalias, fresh-process reload and agreement  3 
 Clear README, provenance and explanation of limitations  1 
 Total  10 

No prescribed accuracy target is required. Correct tracking and evidence matter more than a higher score.

## If you get stuck

 Symptom  First check 
------
 Connection refused  Is the server still running Does the client use the same port 
 UI opens but your run is missing  Check the tracking URI and experiment in the training process 
 Port 5000 is occupied  Use 5001 in the server command, browser address and both client scripts 
 Runs disappear after restart  Did you restart from the same project root with the same SQLite path 
 Loading cannot import a transformer  Is your own package installedimportable in the loading environment 
 Loaded model rejects inputs  Compare column names and types with the recorded examplesignature 
 Alias cannot be resolved  Check registry name, alias spelling, assigned version and tracking URI 
 Dataset input is missing  Check that log_input ran inside the intended active run 
 Model file exists but has no recorded runmodel relationship  Check whether you used model logging or only generic artifact logging 

## Optional extensions — only after the core

- Registry automation if you used the UI, automate registration and alias assignment through the API. Optionally register a second candidate version, move the alias, and compare a fresh load with an already-loaded object.
- Run search use `mlflow.search_runs` to export the candidate comparison as CSV instead of transcribing it.
- Autologging try `mlflow.sklearn.autolog` in a separate experiment. Compare what it records with your explicit logging; identify missing project-specific provenance.

CI, SonarQube and deployment are covered separately. This TP establishes the experiment and model records that those later workflows can use.

## Official documentation

Hints target MLflow 3.x; consult documentation matching the version in your lockfile.

- [MLflow CLI server options](httpsmlflow.orgdocslatestapi_referencecli.html)
- [Tracking server and artifact storage](httpsmlflow.orgdocslatestself-hostingarchitecturetracking-server)
- [MLflow scikit-learn API](httpsmlflow.orgdocslatestapi_referencepython_apimlflow.sklearn.html)
- [Model signatures and input examples](httpsmlflow.orgdocslatestmlmodel)
- [Model Registry](httpsmlflow.orgdocslatestmlmodel-registryworkflow)
- [Dataset tracking API](httpsmlflow.orgdocslatestapi_referencepython_apimlflow.data.html)
