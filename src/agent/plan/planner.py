import os
import sys
import logging
import time

# Add RCRank to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
rcrank_path = os.path.join(current_dir, "RCRank")
if rcrank_path not in sys.path:
    sys.path.append(rcrank_path)

import json

import numpy as np
import pandas as pd
from agents import Agent, Runner
from model.modules.QueryFormer.utils import Encoding
from agent.contracts import (
    SMALL_MODEL_COMPONENTS,
    normalize_component_confidence_map,
    normalize_components,
    parse_detection_output,
)
from RCRank.utils.plan_encoding import PlanEncoder

from agent.prompt import (
    build_root_cause_diagnosis_prompt,
    get_root_cause_llm_instructions,
)
from utils.runtime_metrics import (
    extract_run_usage_metrics,
    format_runtime_metrics,
    merge_runtime_metrics,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = True


class Planner:
    def __init__(self, memory_manager, predictor, agent_config, db=None, database_config=None):
        self.memory_manager = memory_manager
        self.predictor = predictor
        self.agent_config = agent_config
        self.db = db
        self.database_config = database_config or {}
        self.device = predictor.device if predictor is not None else "cpu"
        self._database_info_cache = None

        # Initialize encoding components
        self.encoding = Encoding(None, {"NA": 0})

        self.root_cause_model = self.agent_config.get("root_cause_llm_model")

    def encode_plan(self, query_plan):
        df = pd.DataFrame([{"plan_json": (json.dumps({"Plan": query_plan}) if isinstance(query_plan, dict) else query_plan)}])

        plan_encoder = PlanEncoder(df=df, encoding=self.encoding)
        encoded_df = plan_encoder.df

        return encoded_df["json_plan_tensor"].iloc[0]

    def model_predict(self, query_info):
        start_time = time.perf_counter()
        query, plan, timeseries, log = self._build_prediction_input(query_info)

        pred_label_opt, pred_label = self.predictor.predict(query=query, plan=plan, timeseries=timeseries, log=log)

        result = self._extract_top_root_cause(pred_label, pred_label_opt)
        result["pred_label_opt"] = pred_label_opt.cpu().numpy()
        result["runtime_metrics"] = {
            "T_diag_small_model": time.perf_counter() - start_time,
        }
        logger.info(
            "[Runtime][SQL %s] %s",
            getattr(query_info, "query_id", "unknown"),
            format_runtime_metrics(result["runtime_metrics"]),
        )
        return result

    def diagnosis_agent(self, prompt):
        return Runner.run_sync(
            starting_agent=Agent(
                name="RootCauseLLM",
                model=self.root_cause_model,
                instructions=get_root_cause_llm_instructions(),
            ),
            input=prompt,
        )

    def llm_predict(self, query_info, root_causes, state_confidence, tuning_history=None):
        start_time = time.perf_counter()
        llm_usage_metrics = {
            "diag_llm_requests": 0,
            "diag_llm_input_tokens": 0,
            "diag_llm_output_tokens": 0,
            "diag_llm_total_tokens": 0,
        }
        # ε-greedy retrieval structure: {"positive", "negative"}
        # retrieval = self.memory_manager.retrieve_cases(query_info, root_causes)
        # positives = retrieval.get('positive', [])
        # negatives = retrieval.get('negative', [])

        prompt = build_root_cause_diagnosis_prompt(
            query_info,
            root_causes,
            state_confidence=state_confidence,
            tuning_history=tuning_history,
            database_info=self._build_database_info(),
        )
        logger.info(
            "[Prompt][Detection][SQL %s]\n%s",
            getattr(query_info, "query_id", "unknown"),
            prompt,
        )

        run_result = self.diagnosis_agent(prompt)
        usage_metrics = extract_run_usage_metrics(run_result, "diag_llm")
        for key, value in usage_metrics.items():
            llm_usage_metrics[key] += value
        output = str(run_result.final_output).strip()
        print("output: ", output)
        parsed = self.parse_and_validate(output)
        labels, conf_map = parsed if parsed is not None else (None, None)
        # retry
        retry = 0
        while labels is None and retry < 5:
            run_result = self.diagnosis_agent(prompt)
            usage_metrics = extract_run_usage_metrics(run_result, "diag_llm")
            for key, value in usage_metrics.items():
                llm_usage_metrics[key] += value
            output = str(run_result.final_output).strip()
            parsed = self.parse_and_validate(output)
            labels, conf_map = parsed if parsed is not None else (None, None)
            retry += 1

        if labels is None:
            labels = ["normal"]
            conf_map = {"normal": 1.0}

        runtime_metrics = {
            "T_diag_llm": time.perf_counter() - start_time,
            "diag_llm_retries": retry,
        }
        runtime_metrics = merge_runtime_metrics(runtime_metrics, llm_usage_metrics)
        logger.info(
            "[Runtime][SQL %s] %s",
            getattr(query_info, "query_id", "unknown"),
            format_runtime_metrics(runtime_metrics),
        )
        return {"root_causes": labels, "confidence": conf_map, "runtime_metrics": runtime_metrics}

    def _build_database_info(self):
        if self._database_info_cache is not None:
            return self._database_info_cache

        db_type = getattr(self.db, "db_type", None) or self.database_config.get("db_type") or "unknown"
        workload_type = self.database_config.get("workload_type") or "unknown"
        dbname = self.database_config.get("dbname") or "unknown"
        size = self.db.get_size()[0] or "unknown"
        schema = dbname or "unknown"

        self._database_info_cache = {
            "db_type": db_type,
            "workload_type": workload_type,
            "dbname": dbname,
            "size": size,
            "schema": schema
        }
        return self._database_info_cache

    def parse_and_validate(self, json_text):
        return parse_detection_output(json_text)

    def _build_prediction_input(self, query_info):

        query = query_info.query if hasattr(query_info, "query") else ""
        plan = self.encode_plan(query_info.plan_json)
        # timeseries = json.loads(query_info.external_metrics)
        # log = json.loads(query_info.internal_metrics)
        timeseries = query_info.external_metrics
        log = query_info.internal_metrics

        return query, plan, timeseries, log

    def _extract_top_root_cause(self, pred_label, pred_label_opt):
        pred_label = pred_label.cpu().numpy()
        pred_label_opt = pred_label_opt.cpu().numpy()

        # issue type mapping
        issue_types = {
            0: "index",
            1: "hint",
            2: "knob",
            3: "query",
            4: "normal",
        }

        print("pred_label: ", pred_label)
        print("pred_label_opt: ", pred_label_opt)

        if pred_label.sum() == 0:
            return {"root_causes": "normal"}

        predicted_classes = np.where(pred_label[0] == 1)[0]
        if len(predicted_classes) == 0:
            return {"root_causes": "normal"}
            # return {"root_cause": "normal", "confidence": pred_label_opt}

        root_causes = []
        root_causes_confidence = []
        for class_idx in predicted_classes:
            root_cause = issue_types.get(class_idx)
            if root_cause not in root_causes:
                root_causes.append(root_cause)
                root_causes_confidence.append(pred_label_opt[0][class_idx])

        return {"root_causes": root_causes}
        # root_cause = {
        #     "root_cause": root_causes,
        #     "confidence": root_causes_confidence
        # }

    def predict(self, query_info, state, memory_manager):
        runtime_metrics = {}
        # If current_root already exists, reuse it; otherwise call small model + LLM to get new root cause
        current_root = state.get("current_root")
        if current_root is None:
            # Small model initial prediction + confidence
            model_out = self.model_predict(query_info)
            runtime_metrics = merge_runtime_metrics(runtime_metrics, model_out.get("runtime_metrics"))
            current_root = model_out.get("root_causes")
            print("Small model predicted root cause: ", current_root)
            pred_opt = model_out.get("pred_label_opt")
            labels = list(SMALL_MODEL_COMPONENTS)
            confidence = np.array(pred_opt).reshape(-1)
            state["confidence"] = {labels[i]: float(confidence[i]) for i in range(confidence.shape[0])}
        else:
            current_root = normalize_components(current_root)
            state["confidence"] = normalize_component_confidence_map(state.get("confidence"))

        # Pass current root cause and confidence to LLM
        # Build historical success/failure counts aggregated by root cause
        tuning_history = []
        union_keys = set(state.get("attempts", {}).keys()) | set(state.get("successes", {}).keys())
        for k in union_keys:
            # Normalize to frozenset or single element set
            label = list(k)
            tuning_history.append(
                {
                    "label": label,
                    "success_count": int(state.get("successes", {}).get(k, 0)),
                    "fail_count": int(state.get("attempts", {}).get(k, 0)),
                }
            )

        llm_result = self.llm_predict(
            query_info,
            current_root,
            state_confidence=state.get("confidence"),
            tuning_history=tuning_history,
        )
        runtime_metrics = merge_runtime_metrics(runtime_metrics, llm_result.get("runtime_metrics"))
        predicted_root = llm_result.get("root_causes")
        state["confidence"] = llm_result.get("confidence")
        state["last_runtime_metrics"] = runtime_metrics
        logger.info(
            "[Runtime][SQL %s] Diagnosis summary: %s",
            getattr(query_info, "query_id", "unknown"),
            format_runtime_metrics(runtime_metrics),
        )

        return predicted_root, state
