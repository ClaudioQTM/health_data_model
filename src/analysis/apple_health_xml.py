from __future__ import annotations

from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd


DATE_COLUMNS = {
    "creationDate",
    "startDate",
    "endDate",
    "date",
    "dateComponents",
}

NUMERIC_COLUMNS = {
    "value",
    "duration",
    "totalDistance",
    "totalEnergyBurned",
    "activeEnergyBurned",
    "activeEnergyBurnedGoal",
    "appleMoveTime",
    "appleMoveTimeGoal",
    "appleExerciseTime",
    "appleExerciseTimeGoal",
    "appleStandHours",
    "appleStandHoursGoal",
}

PREFIXES = (
    "HKQuantityTypeIdentifier",
    "HKCategoryTypeIdentifier",
    "HKWorkoutActivityType",
    "HKDataType",
    "HKCharacteristicTypeIdentifier",
)


def short_health_name(value: object) -> object:
    """Remove common HealthKit prefixes while preserving non-string values."""
    if not isinstance(value, str):
        return value

    for prefix in PREFIXES:
        if value.startswith(prefix):
            return value.removeprefix(prefix)
    return value


def _postprocess(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    for column in DATE_COLUMNS.intersection(df.columns):
        df[column] = pd.to_datetime(df[column], errors="coerce")

    numeric_columns = NUMERIC_COLUMNS.intersection(df.columns)
    numeric_columns = numeric_columns.union(
        {column for column in df.columns if column.startswith("stat_")}
    )

    for column in numeric_columns:
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().any():
            df[f"{column}_numeric"] = numeric

    for column in ("type", "workoutActivityType"):
        if column in df.columns:
            df[f"{column}_short"] = df[column].map(short_health_name)

    return df


def _metadata_columns(elem: ET.Element) -> dict[str, str]:
    metadata = {}
    for child in elem:
        if child.tag != "MetadataEntry":
            continue

        key = child.attrib.get("key")
        value = child.attrib.get("value")
        if key:
            metadata[f"metadata_{short_health_name(key)}"] = value

    return metadata


def _workout_child_columns(elem: ET.Element) -> dict[str, object]:
    row: dict[str, object] = {}
    event_count = 0

    for child in elem:
        if child.tag == "WorkoutEvent":
            event_count += 1
            continue

        if child.tag == "WorkoutRoute":
            for route_child in child:
                if route_child.tag == "FileReference":
                    row["route_path"] = route_child.attrib.get("path")
            continue

        if child.tag != "WorkoutStatistics":
            continue

        stat_type = short_health_name(child.attrib.get("type", "unknown"))
        for key, value in child.attrib.items():
            if key in {"type", "startDate", "endDate"}:
                continue
            row[f"stat_{stat_type}_{key}"] = value

    if event_count:
        row["workout_event_count"] = event_count

    return row


def apple_health_to_df(
    xml_path: str | Path,
    tag: str = "Record",
    *,
    type_filter: str | list[str] | set[str] | tuple[str, ...] | None = None,
    include_metadata: bool = False,
    limit: int | None = None,
) -> pd.DataFrame:
    """Stream Apple Health XML elements into a pandas DataFrame.

    Parameters
    ----------
    xml_path:
        Path to Apple Health's ``export.xml``.
    tag:
        Element to load, commonly ``"Record"``, ``"Workout"``, or
        ``"ActivitySummary"``.
    type_filter:
        Optional HealthKit type or collection of types. For ``Record`` this
        matches the ``type`` attribute; for ``Workout`` it matches
        ``workoutActivityType``.
    include_metadata:
        Add ``MetadataEntry`` children as ``metadata_*`` columns.
    limit:
        Optional maximum number of matching rows, useful while exploring.
    """
    xml_path = Path(xml_path)

    if isinstance(type_filter, str):
        allowed_types = {type_filter}
    elif type_filter is None:
        allowed_types = None
    else:
        allowed_types = set(type_filter)

    rows = []
    parsed = ET.iterparse(xml_path, events=("end",))

    child_tags_to_keep = {
        "Record": {
            "MetadataEntry",
            "HeartRateVariabilityMetadataList",
            "InstantaneousBeatsPerMinute",
        },
        "Workout": {
            "MetadataEntry",
            "WorkoutEvent",
            "WorkoutRoute",
            "WorkoutStatistics",
            "FileReference",
        },
    }.get(tag, set())

    for _, elem in parsed:
        if elem.tag != tag:
            if elem.tag not in child_tags_to_keep:
                elem.clear()
            continue

        row = dict(elem.attrib)
        element_type = row.get("type") or row.get("workoutActivityType")
        if allowed_types is not None and element_type not in allowed_types:
            elem.clear()
            continue

        if include_metadata:
            row.update(_metadata_columns(elem))

        if tag == "Workout":
            row.update(_workout_child_columns(elem))

        rows.append(row)
        elem.clear()

        if limit is not None and len(rows) >= limit:
            break

    return _postprocess(pd.DataFrame(rows))


def apple_health_type_counts(
    xml_path: str | Path,
    tag: str = "Record",
) -> pd.DataFrame:
    """Count Apple Health element types without materializing all rows."""
    type_column = "workoutActivityType" if tag == "Workout" else "type"
    counts: Counter[str] = Counter()

    for _, elem in ET.iterparse(Path(xml_path), events=("end",)):
        if elem.tag == tag:
            counts[elem.attrib.get(type_column, "unknown")] += 1
            elem.clear()
        elif tag == "ActivitySummary":
            elem.clear()

    df = pd.DataFrame(
        [{"type": key, "count": value} for key, value in counts.items()]
    ).sort_values("count", ascending=False, ignore_index=True)
    df["type_short"] = df["type"].map(short_health_name)
    return df
