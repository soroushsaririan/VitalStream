"""
infra/sagemaker_pipelines/training_pipeline.py

SageMaker Pipeline definition for the ECG anomaly detection model.

Pipeline stages:
  1. ProcessingJob  — data preparation and train/val/test split
  2. TrainingJob    — 1D-CNN + BiLSTM training with Focal Loss
  3. ProcessingJob  — clinical metric evaluation
  4. ConditionStep  — gate on evaluation report (gates_passed == True)
  5. RegisterModel  — registers approved artifact in Model Registry
     (requires manual approval before deployment proceeds)

Usage:
  python training_pipeline.py --run           # submit pipeline run
  python training_pipeline.py --describe      # print pipeline JSON definition
  python training_pipeline.py --run --wait    # block until complete

Requires: sagemaker>=2.214, boto3>=1.34
"""

from __future__ import annotations

import argparse
import json
import os

import boto3
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.pytorch import PyTorch, PyTorchProcessor
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.model import Model

# ---------------------------------------------------------------------------
# Session and role
# ---------------------------------------------------------------------------

_session = sagemaker.Session()
_region  = boto3.Session().region_name
_account = boto3.client("sts").get_caller_identity()["Account"]

ROLE_ARN        = os.environ["SAGEMAKER_ROLE_ARN"]
MODEL_PKG_GROUP = "ecg-anomaly-detector"
ECR_IMAGE_URI   = f"{_account}.dkr.ecr.{_region}.amazonaws.com/vitalstream-training:latest"

# ---------------------------------------------------------------------------
# Pipeline parameters (overridable at run time without re-defining the pipeline)
# ---------------------------------------------------------------------------

p_train_data   = ParameterString(name="TrainDataS3",  default_value="s3://telemetry-lake/processed/train/")
p_val_data     = ParameterString(name="ValDataS3",    default_value="s3://telemetry-lake/processed/val/")
p_test_data    = ParameterString(name="TestDataS3",   default_value="s3://telemetry-lake/processed/test/")
p_model_ver    = ParameterString(name="ModelVersion", default_value="0.0.1")
p_epochs       = ParameterInteger(name="Epochs",      default_value=30)
p_batch_size   = ParameterInteger(name="BatchSize",   default_value=64)
p_lr           = ParameterFloat(name="LearningRate",  default_value=1e-3)
p_focal_gamma  = ParameterFloat(name="FocalGamma",    default_value=2.0)
p_lstm_hidden  = ParameterInteger(name="LSTMHidden",  default_value=128)


# ---------------------------------------------------------------------------
# Step 1: Data preparation
# ---------------------------------------------------------------------------

def make_data_prep_step() -> ProcessingStep:
    processor = PyTorchProcessor(
        framework_version="2.2",
        py_version="py310",
        role=ROLE_ARN,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        sagemaker_session=_session,
        volume_size_in_gb=50,
        env={"AWS_DEFAULT_REGION": _region},
    )

    return ProcessingStep(
        name="DataPreparation",
        processor=processor,
        inputs=[
            ProcessingInput(source=p_train_data, destination="/opt/ml/processing/input/train"),
            ProcessingInput(source=p_val_data,   destination="/opt/ml/processing/input/val"),
            ProcessingInput(source=p_test_data,  destination="/opt/ml/processing/input/test"),
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/output/train"),
            ProcessingOutput(output_name="val",   source="/opt/ml/processing/output/val"),
            ProcessingOutput(output_name="test",  source="/opt/ml/processing/output/test"),
        ],
        code="src/models/ecg_anomaly/data_prep.py",
    )


# ---------------------------------------------------------------------------
# Step 2: Model training
# ---------------------------------------------------------------------------

def make_training_step(data_prep_step: ProcessingStep) -> TrainingStep:
    estimator = PyTorch(
        entry_point="train.py",
        source_dir="src/models/ecg_anomaly",
        role=ROLE_ARN,
        framework_version="2.2",
        py_version="py310",
        instance_type="ml.g4dn.2xlarge",
        instance_count=1,
        sagemaker_session=_session,
        hyperparameters={
            "epochs":       p_epochs,
            "batch-size":   p_batch_size,
            "lr":           p_lr,
            "focal-gamma":  p_focal_gamma,
            "lstm-hidden":  p_lstm_hidden,
            "model-version": p_model_ver,
        },
        metric_definitions=[
            {"Name": "val_loss",     "Regex": r"val_loss: ([0-9\.]+)"},
            {"Name": "val_accuracy", "Regex": r"val_accuracy: ([0-9\.]+)"},
        ],
        checkpoint_s3_uri=f"s3://telemetry-models/checkpoints/",
        enable_sagemaker_metrics=True,
        volume_size=50,
        max_run=3 * 3600,
    )

    return TrainingStep(
        name="TrainECGModel",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=data_prep_step.properties.ProcessingOutputConfig
                        .Outputs["train"].S3Output.S3Uri
            ),
            "val": TrainingInput(
                s3_data=data_prep_step.properties.ProcessingOutputConfig
                        .Outputs["val"].S3Output.S3Uri
            ),
        },
    )


# ---------------------------------------------------------------------------
# Step 3: Evaluation
# ---------------------------------------------------------------------------

def make_evaluation_step(
    data_prep_step: ProcessingStep,
    training_step: TrainingStep,
) -> ProcessingStep:
    evaluator = PyTorchProcessor(
        framework_version="2.2",
        py_version="py310",
        role=ROLE_ARN,
        instance_type="ml.c5.2xlarge",   # same instance type as the serving endpoint
        instance_count=1,
        sagemaker_session=_session,
    )

    return ProcessingStep(
        name="EvaluateECGModel",
        processor=evaluator,
        inputs=[
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=data_prep_step.properties.ProcessingOutputConfig
                        .Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/output",
                destination=f"s3://telemetry-models/evaluation/",
            )
        ],
        code="src/models/ecg_anomaly/evaluate.py",
    )


# ---------------------------------------------------------------------------
# Step 4 & 5: Condition gate + Model registration
# ---------------------------------------------------------------------------

def make_register_step(
    training_step: TrainingStep,
    evaluation_step: ProcessingStep,
) -> ConditionStep:
    model = Model(
        image_uri=ECR_IMAGE_URI,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        role=ROLE_ARN,
        sagemaker_session=_session,
    )

    register_step = ModelStep(
        name="RegisterECGModel",
        step_args=model.register(
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=["ml.c5.2xlarge", "ml.c5.4xlarge"],
            transform_instances=["ml.c5.2xlarge"],
            model_package_group_name=MODEL_PKG_GROUP,
            approval_status="PendingManualApproval",
            model_metrics=sagemaker.model_metrics.ModelMetrics(
                model_statistics=sagemaker.model_metrics.MetricsSource(
                    s3_uri=evaluation_step.properties.ProcessingOutputConfig
                           .Outputs["evaluation"].S3Output.S3Uri
                           + "/evaluation_report.json",
                    content_type="application/json",
                )
            ),
        ),
    )

    evaluation_report = JsonGet(
        step_name=evaluation_step.name,
        property_file="evaluation_report.json",
        json_path="gates_passed",
    )

    return ConditionStep(
        name="CheckClinicalGates",
        conditions=[ConditionEquals(left=evaluation_report, right=True)],
        if_steps=[register_step],
        else_steps=[],  # Pipeline simply stops; no artifact registered
    )


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    data_prep_step = make_data_prep_step()
    training_step  = make_training_step(data_prep_step)
    eval_step      = make_evaluation_step(data_prep_step, training_step)
    condition_step = make_register_step(training_step, eval_step)

    return Pipeline(
        name="VitalStreamECGTrainingPipeline",
        parameters=[
            p_train_data, p_val_data, p_test_data,
            p_model_ver, p_epochs, p_batch_size,
            p_lr, p_focal_gamma, p_lstm_hidden,
        ],
        steps=[data_prep_step, training_step, eval_step, condition_step],
        sagemaker_session=_session,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",      action="store_true", help="Submit a pipeline execution")
    parser.add_argument("--wait",     action="store_true", help="Block until execution completes")
    parser.add_argument("--describe", action="store_true", help="Print pipeline JSON definition")
    args = parser.parse_args()

    pipeline = build_pipeline()
    pipeline.upsert(role_arn=ROLE_ARN)

    if args.describe:
        print(json.dumps(json.loads(pipeline.definition()), indent=2))

    if args.run:
        execution = pipeline.start()
        print(f"Pipeline execution started: {execution.arn}")
        if args.wait:
            execution.wait()
            print("Pipeline execution complete.")
            print(json.dumps(execution.list_steps(), indent=2, default=str))


if __name__ == "__main__":
    main()
