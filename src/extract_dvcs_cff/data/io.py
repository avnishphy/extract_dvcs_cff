"""
Data ingestion and parsing for DVCS/GPD/CFF datasets.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from .dataset_registry import DatasetRegistry
from .schemas import DatasetRecord, KinematicPoint, ObservableRecord
from .validation import validate_dataset_record

logger = logging.getLogger(__name__)


class BaseDatasetParser:
    """Base class for dataset parsers."""

    def parse(self, file_path: Path) -> DatasetRecord:
        raise NotImplementedError


class OpenGPDTableParser(BaseDatasetParser):
    """Parser for simple table-like datasets (CSV, TSV)."""

    def parse(self, file_path: Path) -> DatasetRecord:
        sep = "\t" if file_path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(file_path, sep=sep)

        required = {"experiment_name", "dataset_id", "observable_name", "value", "xB", "Q2", "t"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in {file_path}: {sorted(missing)}")

        dataset_ids = df["dataset_id"].dropna().unique().tolist()
        if len(dataset_ids) != 1:
            raise ValueError(
                f"Expected exactly one dataset_id in {file_path}, found {len(dataset_ids)}. "
                "Split into separate files or use JSON/YAML payload format."
            )

        experiment_names = df["experiment_name"].dropna().unique().tolist()
        if len(experiment_names) != 1:
            raise ValueError(
                f"Expected exactly one experiment_name in {file_path}, found {len(experiment_names)}."
            )

        publication = str(df["publication"].iloc[0]) if "publication" in df.columns else None
        comments = str(df["comments"].iloc[0]) if "comments" in df.columns else None

        kinematics: List[KinematicPoint] = []
        observables: List[ObservableRecord] = []

        for row in df.itertuples(index=False):
            row_dict = row._asdict()
            phi = _to_float(row_dict.get("phi"))
            if phi is None:
                phi = 0.0

            units: Dict[str, str] = {}
            if "xB_unit" in row_dict and pd.notna(row_dict["xB_unit"]):
                units["xB"] = str(row_dict["xB_unit"])
            if "Q2_unit" in row_dict and pd.notna(row_dict["Q2_unit"]):
                units["Q2"] = str(row_dict["Q2_unit"])
            if "t_unit" in row_dict and pd.notna(row_dict["t_unit"]):
                units["t"] = str(row_dict["t_unit"])
            if "phi_unit" in row_dict and pd.notna(row_dict["phi_unit"]):
                units["phi"] = str(row_dict["phi_unit"])

            kinematics.append(
                KinematicPoint(
                    xB=float(row_dict["xB"]),
                    Q2=float(row_dict["Q2"]),
                    t=float(row_dict["t"]),
                    phi=float(phi),
                    beam_energy=_to_float(row_dict.get("beam_energy")),
                    y=_to_float(row_dict.get("y")),
                    units=units,
                )
            )

            stat_error = _to_float(row_dict.get("stat_error"))
            sys_error = _to_float(row_dict.get("sys_error"))
            total_error = _to_float(row_dict.get("total_error"))
            if total_error is None:
                total_error = _quadrature([stat_error, sys_error])

            observables.append(
                ObservableRecord(
                    observable_name=str(row_dict["observable_name"]),
                    value=float(row_dict["value"]),
                    stat_error=stat_error,
                    sys_error=sys_error,
                    total_error=total_error,
                    covariance_id=(
                        str(row_dict["covariance_id"])
                        if "covariance_id" in row_dict and pd.notna(row_dict["covariance_id"])
                        else None
                    ),
                    channel=(
                        str(row_dict["channel"])
                        if "channel" in row_dict and pd.notna(row_dict["channel"])
                        else None
                    ),
                )
            )

        record = DatasetRecord(
            experiment_name=experiment_names[0],
            dataset_id=dataset_ids[0],
            publication=publication,
            observables=observables,
            kinematics=kinematics,
            comments=comments,
        )
        return record


class JSONDatasetParser(BaseDatasetParser):
    def parse(self, file_path: Path) -> DatasetRecord:
        with open(file_path, "r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
        return _parse_dataset_payload(payload, source=file_path)


class YAMLDatasetParser(BaseDatasetParser):
    def parse(self, file_path: Path) -> DatasetRecord:
        with open(file_path, "r", encoding="utf-8") as file_handle:
            payload = yaml.safe_load(file_handle)
        return _parse_dataset_payload(payload, source=file_path)


PARSER_MAP = {
    ".csv": OpenGPDTableParser(),
    ".tsv": OpenGPDTableParser(),
    ".json": JSONDatasetParser(),
    ".yaml": YAMLDatasetParser(),
    ".yml": YAMLDatasetParser(),
}


def _parse_dataset_payload(payload: Any, source: Path) -> DatasetRecord:
    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError(
                f"Expected one DatasetRecord in {source}, found list with {len(payload)} entries."
            )
        payload = payload[0]

    if not isinstance(payload, dict):
        raise ValueError(f"Dataset payload in {source} must be a mapping/dictionary.")

    return DatasetRecord.model_validate(payload)


def _safe_call(callable_obj, default=None):
    try:
        return callable_obj()
    except Exception:
        return default


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        cast = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(cast):
        return None
    return cast


def _quadrature(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not clean:
        return None
    return float(np.sqrt(np.sum(np.square(clean))))


def _symmetrized_uncertainty(unc_obj) -> Optional[float]:
    if unc_obj is None:
        return None

    try:
        if unc_obj.is_asymmetric():
            low = _to_float(unc_obj.get_unc_lower())
            high = _to_float(unc_obj.get_unc_upper())
            if low is None or high is None:
                return None
            return 0.5 * (low + high)
        return _to_float(unc_obj.get_unc())
    except Exception:
        return None


def _uncertainty_value(unc_set, index: int) -> Optional[float]:
    if unc_set is None:
        return None
    try:
        unc_obj = unc_set.get_uncertainty(index)
    except Exception:
        return None
    return _symmetrized_uncertainty(unc_obj)


def _correlation_label(unc_set) -> Optional[str]:
    if unc_set is None:
        return None
    label = _safe_call(unc_set.get_correlation_matrix)
    if label is None:
        return None
    return str(label)


def _normalize_gpddatabase_root(path_like: Path) -> Path:
    root = Path(path_like).expanduser().resolve()
    if (root / "gpddatabase" / "data").is_dir():
        return root
    if (root / "data").is_dir() and (root / "ExclusiveDatabase.py").is_file():
        return root.parent
    raise ValueError(
        f"Invalid gpddatabase_root: {root}. Expected gpddatabase repository root containing gpddatabase/data."
    )


def _import_local_gpddatabase(repo_root: Path):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        return importlib.import_module("gpddatabase")
    except ModuleNotFoundError as exc:
        raise ValueError(
            f"Could not import gpddatabase from {repo_root}. "
            "Ensure the repository exists and contains the gpddatabase package."
        ) from exc


def _configure_gpddatabase_paths(db, repo_root: Path) -> None:
    data_root = repo_root / "gpddatabase" / "data"
    candidates = [
        data_root / "DVCS",
        data_root / "latticeQCD",
        data_root / "structure_function",
        data_root / "other",
    ]
    existing = [str(path) for path in candidates if path.is_dir()]
    if not existing:
        raise ValueError(f"No gpddatabase data directories found under {data_root}")

    db.set_path_to_databse(":".join(existing))


def _extract_beam_energy(conditions: Any) -> Optional[float]:
    if not isinstance(conditions, dict):
        return None

    for key in ("lepton_beam_energy", "beam_energy", "hadron_beam_energy"):
        if key in conditions:
            value = _to_float(conditions.get(key))
            if value is not None and value > 0:
                return value
    return None


def _load_gpddatabase_records(config) -> List[DatasetRecord]:
    ingestion = config.ingestion
    gpddb_root = getattr(ingestion, "gpddatabase_root", None)
    if gpddb_root is None:
        return []

    repo_root = _normalize_gpddatabase_root(Path(gpddb_root))
    gpddb = _import_local_gpddatabase(repo_root)
    db = gpddb.ExclusiveDatabase()
    _configure_gpddatabase_paths(db, repo_root)

    uuid_filter = getattr(ingestion, "gpddatabase_uuid", None)
    collaboration_filter = getattr(ingestion, "gpddatabase_collaboration", None)
    data_type_filter = getattr(ingestion, "gpddatabase_data_type", None)
    include_pseudodata = bool(getattr(ingestion, "include_pseudodata", False))
    strict_kinematics = bool(getattr(ingestion, "strict_kinematics", True))

    grouped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    skipped_points = 0

    for uuid in db.get_uuids():
        if uuid_filter and uuid != uuid_filter:
            continue

        data_object = db.get_data_object(uuid)
        meta = data_object.get_general_info()

        collaboration = _safe_call(meta.get_collaboration)
        data_type = _safe_call(meta.get_data_type)
        pseudodata = bool(_safe_call(meta.get_pseudodata, False))
        reference = _safe_call(meta.get_reference)
        comment = _safe_call(meta.get_comment)
        conditions = _safe_call(meta.get_conditions, {})

        if collaboration_filter and collaboration != collaboration_filter:
            continue
        if data_type_filter and data_type != data_type_filter:
            continue
        if not include_pseudodata and pseudodata:
            continue

        data = data_object.get_data()
        for dataset_label in data.get_data_set_labels():
            dataset = data.get_data_set(dataset_label)

            for point_index in range(dataset.get_number_of_data_points()):
                point = dataset.get_data_point(point_index)

                kin_names = list(point.get_kinematics_names() or [])
                kin_values = list(point.get_kinematics_values() or [])
                kin_units = list(point.get_kinematics_units() or [])

                kin_map = {str(name): _to_float(value) for name, value in zip(kin_names, kin_values)}
                unit_map = {
                    str(name): (None if unit is None else str(unit))
                    for name, unit in zip(kin_names, kin_units)
                }

                xB = kin_map.get("xB")
                Q2 = kin_map.get("Q2")
                t = kin_map.get("t")
                phi = kin_map.get("phi")

                if xB is None or Q2 is None or t is None:
                    skipped_points += 1
                    continue

                if phi is None:
                    # Keep compatibility with schemas that expect a value.
                    phi = 0.0

                beam_energy = _extract_beam_energy(conditions)
                y = kin_map.get("y")

                try:
                    kinematics = KinematicPoint(
                        xB=float(xB),
                        Q2=float(Q2),
                        t=float(t),
                        phi=float(phi),
                        beam_energy=beam_energy,
                        y=y,
                        units={k: v for k, v in unit_map.items() if v is not None},
                    )
                except ValueError:
                    skipped_points += 1
                    if strict_kinematics:
                        continue

                    # Relaxed mode: sanitize mildly unphysical values into valid bounds.
                    xB_relaxed = min(max(float(xB), 1e-8), 1.0 - 1e-8)
                    Q2_relaxed = max(float(Q2), 1e-8)
                    t_relaxed = -abs(float(t))
                    phi_relaxed = float(phi) % 360.0
                    y_relaxed = None
                    if y is not None:
                        y_relaxed = min(max(float(y), 0.0), 1.0)

                    beam_energy_relaxed = beam_energy
                    if beam_energy_relaxed is not None and beam_energy_relaxed <= 0.0:
                        beam_energy_relaxed = None

                    try:
                        kinematics = KinematicPoint(
                            xB=xB_relaxed,
                            Q2=Q2_relaxed,
                            t=t_relaxed,
                            phi=phi_relaxed,
                            beam_energy=beam_energy_relaxed,
                            y=y_relaxed,
                            units={k: v for k, v in unit_map.items() if v is not None},
                        )
                    except ValueError:
                        continue

                obs_names = list(point.get_observables_names() or [])
                obs_values = list(point.get_observables_values() or [])
                obs_len = min(len(obs_names), len(obs_values))

                stat_set = point.get_observables_stat_uncertainties()
                sys_set = point.get_observables_sys_uncertainties()
                norm_set = point.get_observables_norm_uncertainties()

                for obs_index in range(obs_len):
                    obs_name = str(obs_names[obs_index])
                    value = _to_float(obs_values[obs_index])
                    if value is None:
                        skipped_points += 1
                        continue

                    stat_error = _uncertainty_value(stat_set, obs_index)
                    sys_error = _uncertainty_value(sys_set, obs_index)
                    norm_error = _uncertainty_value(norm_set, obs_index)
                    total_error = _quadrature([stat_error, sys_error, norm_error])

                    covariance_id = (
                        _correlation_label(stat_set)
                        or _correlation_label(sys_set)
                        or _correlation_label(norm_set)
                    )

                    key = (uuid, dataset_label, obs_name)
                    if key not in grouped:
                        grouped[key] = {
                            "experiment_name": collaboration or "unknown",
                            "dataset_id": f"{uuid}:{dataset_label}:{obs_name}",
                            "publication": reference,
                            "comments": comment,
                            "kinematics": [],
                            "observables": [],
                        }

                    grouped[key]["kinematics"].append(kinematics)
                    grouped[key]["observables"].append(
                        ObservableRecord(
                            observable_name=obs_name,
                            value=float(value),
                            stat_error=stat_error,
                            sys_error=sys_error,
                            total_error=total_error,
                            covariance_id=covariance_id,
                            channel=dataset_label,
                        )
                    )

    records: List[DatasetRecord] = []
    for payload in grouped.values():
        record = DatasetRecord(
            experiment_name=payload["experiment_name"],
            dataset_id=payload["dataset_id"],
            publication=payload["publication"],
            observables=payload["observables"],
            kinematics=payload["kinematics"],
            comments=payload["comments"],
        )
        validate_dataset_record(record)
        records.append(record)

    logger.info(
        "Loaded %d DatasetRecord entries from gpddatabase (skipped %d incompatible points).",
        len(records),
        skipped_points,
    )

    return records


def load_dataset(file_path: Path) -> DatasetRecord:
    ext = file_path.suffix.lower()
    parser = PARSER_MAP.get(ext)
    if parser is None:
        raise ValueError(f"Unsupported file extension: {ext}")

    record = parser.parse(file_path)
    validate_dataset_record(record)
    return record


def load_all_datasets(config) -> List[DatasetRecord]:
    files = list(getattr(config.ingestion, "dataset_files", []))
    registry = DatasetRegistry()
    records: List[DatasetRecord] = []

    for file_path in files:
        record = load_dataset(Path(file_path))
        registry.register(record)
        records.append(record)

    gpddb_records = _load_gpddatabase_records(config)
    for record in gpddb_records:
        registry.register(record)
        records.append(record)

    return records
