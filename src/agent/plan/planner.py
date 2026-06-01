import asyncio
import os
import sys
import logging
import time
from concurrent.futures import ThreadPoolExecutor

# Add RCRank to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
rcrank_path = os.path.join(current_dir, "RCRank")
if rcrank_path not in sys.path:
    sys.path.append(rcrank_path)

import json

import numpy as np
import pandas as pd
from agents import Agent, Runner
from model.modules.QueryFormer.utils import Encoding
from agent.contracts import (
    CANONICAL_COMPONENTS,
    normalize_component_confidence_map,
    normalize_components,
    parse_detection_output,
)
from agent.plan.localization import (
    EXPERT_LIGHTWEIGHT,
    EXPERT_LLM,
    EXPERT_RANDOM,
    LocalizationExpertSelector,
    build_tuning_history,
    random_component_subset,
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
        self.localization_selector = LocalizationExpertSelector(agent_config)

        # Initialize encoding components
        self.encoding = Encoding(None, {"NA": 0})

        self.root_cause_model = self.agent_config.get("root_cause_llm_model")

    def _run_agent_sync(self, agent, prompt):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return Runner.run_sync(starting_agent=agent, input=prompt)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(Runner.run_sync, starting_agent=agent, input=prompt)
            return future.result()

    def encode_plan(self, query_plan):
        df = pd.DataFrame(
            [{"plan_json": (json.dumps({"Plan": query_plan}) if isinstance(query_plan, dict) else query_plan)}]
        )

        plan_encoder = PlanEncoder(df=df, encoding=self.encoding)
        encoded_df = plan_encoder.df

        return encoded_df["json_plan_tensor"].iloc[0]

    def model_predict(self, query_info):
        if self.predictor is None:
            raise RuntimeError("Lightweight localization model is not configured")
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
        agent = Agent(
            name="RootCauseLLM",
            model=self.root_cause_model,
            instructions=get_root_cause_llm_instructions(),
        )
        return self._run_agent_sync(agent, prompt)

    def llm_predict(self, query_info, tuning_history=None):
        start_time = time.perf_counter()
        llm_usage_metrics = {
            "diag_llm_requests": 0,
            "diag_llm_input_tokens": 0,
            "diag_llm_output_tokens": 0,
            "diag_llm_total_tokens": 0,
        }
        prompt = build_root_cause_diagnosis_prompt(
            query_info,
            tuning_history=tuning_history,
            database_info=self._build_database_info(),
        )
        # logger.info(
        #     "[Prompt][Detection][SQL %s]\n%s",
        #     getattr(query_info, "query_id", "unknown"),
        #     prompt,
        # )

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
            labels = random_component_subset()
            conf_map = {component: (0.6 if component in labels else 0.0) for component in CANONICAL_COMPONENTS}

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
            "schema": schema,
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
        }

        confidence_map = {
            issue_types[class_idx]: float(pred_label_opt[0][class_idx])
            for class_idx in range(min(pred_label_opt.shape[1], len(issue_types)))
        }

        predicted_classes = np.where(pred_label[0] == 1)[0]
        if len(predicted_classes) == 0:
            best_idx = int(np.argmax(pred_label_opt[0][: len(issue_types)]))
            predicted_classes = np.array([best_idx])

        root_causes = []
        for class_idx in predicted_classes:
            root_cause = issue_types.get(class_idx)
            if root_cause not in root_causes:
                root_causes.append(root_cause)

        return {"root_causes": root_causes, "confidence": confidence_map}

    def _allowed_localization_experts(self):
        model_type = str(self.agent_config.get("model_type", "hybrid")).lower()
        if model_type in {"small", "small_model", "lm"}:
            return [EXPERT_LIGHTWEIGHT] if self.predictor is not None else [EXPERT_RANDOM]
        if model_type in {"large", "large_model", "llm"}:
            return [EXPERT_LLM]
        if model_type in {"random", "rand"}:
            return [EXPERT_RANDOM]
        experts = []
        if self.predictor is not None:
            experts.append(EXPERT_LIGHTWEIGHT)
        experts.extend([EXPERT_LLM, EXPERT_RANDOM])
        return experts

    def _ensure_nonempty_components(self, root_causes, confidence=None):
        root_causes = normalize_components(root_causes)
        if root_causes:
            return root_causes

        confidence = normalize_component_confidence_map(confidence)
        if confidence:
            return [max(CANONICAL_COMPONENTS, key=lambda label: confidence.get(label, 0.0))]
        return random_component_subset()

    def _predict_with_random_expert(self, state):
        exclude_roots = state.get("root_tried", set())
        root_causes = random_component_subset(exclude_roots=exclude_roots)
        confidence = {component: (0.6 if component in root_causes else 0.0) for component in CANONICAL_COMPONENTS}
        return {"root_causes": root_causes, "confidence": confidence, "runtime_metrics": {}}

    def predict(self, query_info, state, memory_manager):
        runtime_metrics = {}
        active_decision = self.localization_selector.active_decision(state)
        if active_decision is not None:
            state["current_root"] = active_decision.root_causes
            state["confidence"] = active_decision.confidence
            state["localization_expert"] = active_decision.expert
            state["last_runtime_metrics"] = runtime_metrics
            return active_decision.root_causes, state

        expert, samples = self.localization_selector.select_expert(state, self._allowed_localization_experts())
        state.setdefault("localization", {})["last_samples"] = samples

        if expert == EXPERT_LIGHTWEIGHT:
            result = self.model_predict(query_info)
        elif expert == EXPERT_LLM:
            result = self.llm_predict(
                query_info,
                tuning_history=build_tuning_history(state),
            )
        else:
            result = self._predict_with_random_expert(state)

        runtime_metrics = merge_runtime_metrics(runtime_metrics, result.get("runtime_metrics"))
        confidence = normalize_component_confidence_map(result.get("confidence"))
        predicted_root = self._ensure_nonempty_components(result.get("root_causes"), confidence)
        predicted_key = frozenset(predicted_root)
        if predicted_key in state.get("root_tried", set()) and expert != EXPERT_RANDOM:
            result = self._predict_with_random_expert(state)
            confidence = normalize_component_confidence_map(result.get("confidence"))
            predicted_root = self._ensure_nonempty_components(result.get("root_causes"), confidence)
            expert = EXPERT_RANDOM
        decision = self.localization_selector.start_decision(state, expert, predicted_root, confidence)
        state["localization_expert"] = decision.expert
        state["last_runtime_metrics"] = runtime_metrics
        logger.info(
            "[Runtime][SQL %s] Localization summary: expert=%s components=%s %s",
            getattr(query_info, "query_id", "unknown"),
            expert,
            predicted_root,
            format_runtime_metrics(runtime_metrics),
        )

        return predicted_root, state

    def update_localization_feedback(self, state, success):
        return self.localization_selector.record_feedback(state, success)
