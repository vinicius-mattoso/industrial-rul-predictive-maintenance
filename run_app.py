from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles


ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / "app"
DATA_PATH = ROOT / "data" / "raw" / "train_FD001.txt"
ARTIFACTS_DIR = ROOT / "artifacts"

FEATURE_SENSORS = [
    "sensor_4",
    "sensor_7",
    "sensor_11",
    "sensor_12",
    "sensor_15",
    "sensor_21",
]
ROLLING_WINDOW = 5
INITIAL_CYCLE = 85
FLEET_ENGINE_LIMIT = 18


class MaintenanceDataService:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.model_message = "Modelo nao carregado."
        self.label_mapping: dict[int, str] = {0: "Healthy", 1: "Degrading", 2: "Critical"}
        self.model_features: list[str] = []
        self.df = self._load_dataset()
        self._load_artifacts()

    def _load_dataset(self) -> pd.DataFrame:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

        columns = (
            ["unit_number", "time_in_cycles", "operational_setting_1", "operational_setting_2", "operational_setting_3"]
            + [f"sensor_{i}" for i in range(1, 22)]
        )
        df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, names=columns)
        max_cycles = df.groupby("unit_number")["time_in_cycles"].transform("max")
        df["RUL"] = max_cycles - df["time_in_cycles"]
        df["RUL_capped"] = df["RUL"].clip(upper=125)
        df["health_state_rule"] = df["RUL_capped"].apply(self._rule_status)

        for sensor in FEATURE_SENSORS:
            grouped = df.groupby("unit_number")[sensor]
            df[f"{sensor}_rolling_mean"] = grouped.transform(
                lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
            )
            df[f"{sensor}_rolling_std"] = grouped.transform(
                lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).std()
            )
            df[f"{sensor}_delta"] = grouped.diff()
            df[f"{sensor}_trend"] = grouped.diff(periods=3)

        engineered = [column for column in df.columns if "rolling" in column or "delta" in column or "trend" in column]
        df[engineered] = df[engineered].fillna(0)
        return df

    def _load_artifacts(self) -> None:
        try:
            self.model_features = joblib.load(ARTIFACTS_DIR / "model_features.pkl")
            self.label_mapping = joblib.load(ARTIFACTS_DIR / "label_mapping.pkl")
            self.model = joblib.load(ARTIFACTS_DIR / "lightgbm_health_classifier.pkl")
            self.model_message = "Modelo LightGBM carregado dos artefatos."
        except Exception as exc:
            self.model = None
            self.model_message = (
                "Modelo indisponivel no runtime atual. "
                f"Usando regra de RUL como fallback. Detalhe: {exc.__class__.__name__}: {exc}"
            )

    @staticmethod
    def _rule_status(rul: float) -> str:
        if rul > 80:
            return "Healthy"
        if rul > 40:
            return "Degrading"
        return "Critical"

    @staticmethod
    def _ui_status(status: str) -> str:
        if status == "Degrading":
            return "Warning"
        return status

    def _prediction_for_row(self, row: pd.Series) -> tuple[str, float | None]:
        if self.model is None or not self.model_features:
            return self._ui_status(str(row["health_state_rule"])), None

        x = pd.DataFrame([row[self.model_features].to_dict()])
        prediction = int(self.model.predict(x)[0])
        status = self._ui_status(self.label_mapping.get(prediction, str(prediction)))
        confidence = None
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(x)[0]
            confidence = round(float(max(probabilities)), 3)
        return status, confidence

    def engine_ids(self) -> list[int]:
        return [int(engine_id) for engine_id in sorted(self.df["unit_number"].unique())]

    def engine_frame(self, engine_id: int) -> pd.DataFrame:
        return self.df[self.df["unit_number"] == engine_id].copy()

    def row_at_engine_cycle(self, engine_id: int, cycle: int) -> pd.Series:
        engine_df = self.engine_frame(engine_id)
        target_cycle = min(max(cycle, 1), int(engine_df["time_in_cycles"].max()))
        candidates = engine_df[engine_df["time_in_cycles"] <= target_cycle]
        if candidates.empty:
            return engine_df.iloc[0]
        return candidates.iloc[-1]

    def feature_vector_for_row(self, row: pd.Series) -> pd.DataFrame:
        if not self.model_features:
            return pd.DataFrame()
        return pd.DataFrame([row[self.model_features].to_dict()])

    def prediction_details_for_row(self, row: pd.Series) -> dict[str, Any]:
        status, confidence = self._prediction_for_row(row)
        probabilities = None
        if self.model is not None and self.model_features and hasattr(self.model, "predict_proba"):
            x = self.feature_vector_for_row(row)
            probability_values = self.model.predict_proba(x)[0]
            probabilities = {
                self._ui_status(self.label_mapping.get(index, str(index))): round(float(value), 4)
                for index, value in enumerate(probability_values)
            }
        return {
            "status": status,
            "confidence": confidence,
            "probabilities": probabilities,
            "fallbackRuleStatus": self._ui_status(str(row["health_state_rule"])),
        }

    def prediction_details_for_features(self, features: pd.DataFrame, fallback_rul: float) -> dict[str, Any]:
        fallback_status = self._ui_status(self._rule_status(fallback_rul))
        if self.model is None or features.empty:
            return {
                "status": fallback_status,
                "confidence": None,
                "probabilities": None,
                "fallbackRuleStatus": fallback_status,
            }

        prediction = int(self.model.predict(features[self.model_features])[0])
        status = self._ui_status(self.label_mapping.get(prediction, str(prediction)))
        probabilities = None
        confidence = None
        if hasattr(self.model, "predict_proba"):
            probability_values = self.model.predict_proba(features[self.model_features])[0]
            probabilities = {
                self._ui_status(self.label_mapping.get(index, str(index))): round(float(value), 4)
                for index, value in enumerate(probability_values)
            }
            confidence = round(float(max(probability_values)), 3)

        return {
            "status": status,
            "confidence": confidence,
            "probabilities": probabilities,
            "fallbackRuleStatus": fallback_status,
        }

    def simulated_feature_vector(
        self,
        engine_id: int,
        base_cycle: int,
        sensor_values: dict[str, float],
        simulated_rul: float,
    ) -> pd.DataFrame:
        engine_df = self.engine_frame(engine_id)
        history = engine_df[engine_df["time_in_cycles"] <= base_cycle].copy()
        if history.empty:
            history = engine_df.head(1).copy()

        base_row = self.row_at_engine_cycle(engine_id, base_cycle).copy()
        simulated_row = base_row.copy()
        simulated_row["time_in_cycles"] = int(base_cycle) + 1
        simulated_row["RUL"] = float(simulated_rul)
        simulated_row["RUL_capped"] = min(float(simulated_rul), 125.0)
        simulated_row["health_state_rule"] = self._rule_status(float(simulated_row["RUL_capped"]))
        for sensor, value in sensor_values.items():
            simulated_row[sensor] = float(value)

        simulation_df = pd.concat([history, simulated_row.to_frame().T], ignore_index=True)
        numeric_columns = ["time_in_cycles", "RUL", "RUL_capped", *FEATURE_SENSORS]
        simulation_df[numeric_columns] = simulation_df[numeric_columns].apply(pd.to_numeric, errors="coerce")
        for sensor in FEATURE_SENSORS:
            simulation_df[f"{sensor}_rolling_mean"] = (
                simulation_df[sensor].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
            )
            simulation_df[f"{sensor}_rolling_std"] = (
                simulation_df[sensor].rolling(window=ROLLING_WINDOW, min_periods=1).std()
            )
            simulation_df[f"{sensor}_delta"] = simulation_df[sensor].diff()
            simulation_df[f"{sensor}_trend"] = simulation_df[sensor].diff(periods=3)

        feature_columns = [
            column
            for column in simulation_df.columns
            if column in FEATURE_SENSORS or "rolling" in column or "delta" in column or "trend" in column
        ]
        simulation_df[feature_columns] = simulation_df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        last_row = simulation_df.iloc[-1]
        return self.feature_vector_for_row(last_row)

    def critical_sensor_for_row(self, row: pd.Series) -> str:
        return self._critical_sensor(row)

    def _row_at_cycle(self, engine_id: int, offset: int) -> pd.Series:
        engine_df = self.df[self.df["unit_number"] == engine_id]
        target_cycle = min(INITIAL_CYCLE + offset, int(engine_df["time_in_cycles"].max()))
        candidates = engine_df[engine_df["time_in_cycles"] <= target_cycle]
        if candidates.empty:
            return engine_df.iloc[0]
        return candidates.iloc[-1]

    def fleet_snapshot(self, offset: int, selected_engine: int | None) -> dict[str, Any]:
        engine_ids = self.engine_ids()
        visible_ids = engine_ids[:FLEET_ENGINE_LIMIT]
        if selected_engine and selected_engine not in visible_ids:
            visible_ids = [selected_engine] + visible_ids[:-1]

        equipment = [self._equipment_payload(engine_id, offset) for engine_id in visible_ids]
        sensor_engine = selected_engine or visible_ids[0]

        return {
            "cycleOffset": offset,
            "operationCycle": INITIAL_CYCLE + offset,
            "model": {
                "available": self.model is not None,
                "message": self.model_message,
                "features": self.model_features,
            },
            "engines": engine_ids,
            "selectedEngine": sensor_engine,
            "featureSensors": FEATURE_SENSORS,
            "equipment": equipment,
            "sensorSeries": self.sensor_series(sensor_engine, offset),
            "rulHistory": self.rul_history(visible_ids, offset),
        }

    def _equipment_payload(self, engine_id: int, offset: int) -> dict[str, Any]:
        row = self._row_at_cycle(engine_id, offset)
        status, confidence = self._prediction_for_row(row)
        return {
            "id": f"ENGINE-{engine_id:03d}",
            "engineId": engine_id,
            "area": f"Linha {chr(65 + ((engine_id - 1) % 4))}",
            "cycle": int(row["time_in_cycles"]),
            "maxCycle": int(self.df[self.df["unit_number"] == engine_id]["time_in_cycles"].max()),
            "rul": round(float(row["RUL_capped"]), 1),
            "status": status,
            "confidence": confidence,
            "criticalSensor": self._critical_sensor(row),
            "sensors": {sensor: round(float(row[sensor]), 4) for sensor in FEATURE_SENSORS},
        }

    def _critical_sensor(self, row: pd.Series) -> str:
        scores = {}
        for sensor in FEATURE_SENSORS:
            engine_reference = self.df[
                (self.df["unit_number"] == row["unit_number"])
                & (self.df["health_state_rule"] == "Healthy")
            ][sensor]
            if engine_reference.empty:
                scores[sensor] = 0
                continue
            p05 = engine_reference.quantile(0.05)
            p95 = engine_reference.quantile(0.95)
            width = max(float(p95 - p05), 0.001)
            value = float(row[sensor])
            if value < p05:
                scores[sensor] = float((p05 - value) / width)
            elif value > p95:
                scores[sensor] = float((value - p95) / width)
            else:
                scores[sensor] = 0
        return max(scores, key=scores.get)

    def sensor_series(self, engine_id: int, offset: int) -> dict[str, Any]:
        engine_df = self.df[self.df["unit_number"] == engine_id].copy()
        current_cycle = int(self._row_at_cycle(engine_id, offset)["time_in_cycles"])
        window_df = engine_df[engine_df["time_in_cycles"] <= current_cycle].tail(80)

        rows = []
        for _, row in window_df.iterrows():
            rows.append(
                {
                    "cycle": int(row["time_in_cycles"]),
                    "status": self._ui_status(str(row["health_state_rule"])),
                    **{sensor: round(float(row[sensor]), 4) for sensor in FEATURE_SENSORS},
                }
            )

        healthy = engine_df[engine_df["health_state_rule"] == "Healthy"]
        envelope = {}
        for sensor in FEATURE_SENSORS:
            reference = healthy[sensor] if not healthy.empty else engine_df[sensor]
            envelope[sensor] = {
                "mean": round(float(reference.mean()), 4),
                "p05": round(float(reference.quantile(0.05)), 4),
                "p95": round(float(reference.quantile(0.95)), 4),
                "min": round(float(reference.min()), 4),
                "max": round(float(reference.max()), 4),
            }

        return {
            "engineId": engine_id,
            "currentCycle": current_cycle,
            "rows": rows,
            "envelope": envelope,
        }

    def rul_history(self, engine_ids: list[int], offset: int) -> list[dict[str, Any]]:
        start = max(0, offset - 11)
        history = []
        for item_offset in range(start, offset + 1):
            ruls = [self._equipment_payload(engine_id, item_offset)["rul"] for engine_id in engine_ids]
            history.append(
                {
                    "cycle": INITIAL_CYCLE + item_offset,
                    "avg": round(sum(ruls) / len(ruls), 1),
                }
            )
        return history


DATA_SERVICE = MaintenanceDataService()


api = FastAPI(
    title="Industrial RUL Predictive Maintenance API",
    description="Local API for the predictive maintenance control tower.",
    version="1.0.0",
)


@api.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "modelAvailable": DATA_SERVICE.model is not None,
        "modelMessage": DATA_SERVICE.model_message,
    }


@api.get("/api/engines")
def engines() -> dict[str, list[int]]:
    return {"engines": DATA_SERVICE.engine_ids()}


@api.get("/api/snapshot")
def snapshot(
    offset: int = Query(default=0, ge=0),
    engine: int | None = Query(default=None, ge=1),
) -> dict[str, Any]:
    return DATA_SERVICE.fleet_snapshot(offset=offset, selected_engine=engine)


api.mount("/", StaticFiles(directory=str(APP_DIR), html=True), name="static")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local predictive maintenance control tower UI."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind. Default: 8765")
    parser.add_argument("--open", action="store_true", help="Open the dashboard after startup.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not APP_DIR.exists():
        raise SystemExit(f"App directory not found: {APP_DIR}")

    url = f"http://{args.host}:{args.port}"
    print("")
    print("RUL Control Tower is running with FastAPI")
    print(f"URL: {url}")
    print(f"API docs: {url}/docs")
    print(DATA_SERVICE.model_message)
    print("Press Ctrl+C to stop.")
    print("")

    if args.open:
        webbrowser.open(url)

    uvicorn.run(api, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
