import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path



def apple_health_xml_to_df(
    xml_path: str | Path,
    tag: str | None = None,
    *,
    type_filter: str | set[str] | list[str] | tuple[str, ...] | None = None,
    ancestor_type_filter: str | set[str] | list[str] | tuple[str, ...] | None = None,
    include_ancestors: bool = True,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load Apple Health XML elements into a DataFrame using one generic parser."""
    xml_path = Path(xml_path)

    if isinstance(type_filter, str):
        type_filter = {type_filter}
    elif type_filter is not None:
        type_filter = set(type_filter)

    if isinstance(ancestor_type_filter, str):
        ancestor_type_filter = {ancestor_type_filter}
    elif ancestor_type_filter is not None:
        ancestor_type_filter = set(ancestor_type_filter)

    rows = []
    stack = []

    for event, elem in ET.iterparse(xml_path, events=("start", "end")):
        if event == "start":
            stack.append((elem.tag, dict(elem.attrib)))
            continue

        elem_tag = elem.tag
        elem_attrs = dict(elem.attrib)
        ancestors = stack[:-1]

        if tag is not None and elem_tag != tag:
            stack.pop()
            elem.clear()
            continue

        element_type = (
            elem_attrs.get("type")
            or elem_attrs.get("workoutActivityType")
        )

        if type_filter is not None and element_type not in type_filter:
            stack.pop()
            elem.clear()
            continue

        if ancestor_type_filter is not None:
            ancestor_types = {
                attrs.get("type") or attrs.get("workoutActivityType")
                for _, attrs in ancestors
            }
            if not ancestor_types.intersection(ancestor_type_filter):
                stack.pop()
                elem.clear()
                continue

        row = {"element_tag": elem_tag}
        row.update(elem_attrs)

        if include_ancestors:
            for ancestor_tag, ancestor_attrs in ancestors:
                for key, value in ancestor_attrs.items():
                    row[f"parent_{ancestor_tag}_{key}"] = value

        rows.append(row)

        stack.pop()
        elem.clear()

        if limit is not None and len(rows) >= limit:
            break

    df = pd.DataFrame(rows)

    for column in df.columns:
        if column.endswith("Date") or column in {"date", "dateComponents"}:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    for column in ["value", "duration", "bpm", "count"]:
        if column in df.columns:
            df[f"{column}_numeric"] = pd.to_numeric(df[column], errors="coerce")

    return df

__all__ = ["apple_health_xml_to_df"]