# app.py
import streamlit as st
import pandas as pd
import json
import xml.etree.ElementTree as ET
from io import BytesIO, StringIO
import re
from datetime import datetime
from pandas import json_normalize

st.set_page_config(page_title="Auto Schema Feed Validator (JSON / XML)", layout="wide")

# -----------------------
# Utility: read spec CSV
# -----------------------
@st.cache_data
def load_spec_df(spec_file_bytes):
    # try utf-8 then latin1 fallback
    try:
        df = pd.read_csv(spec_file_bytes)
    except Exception:
        spec_file_bytes.seek(0)
        df = pd.read_csv(spec_file_bytes, encoding="latin-1")
    # normalize header names
    df.columns = [c.strip() for c in df.columns]
    return df

# -----------------------
# Utilities: flatten JSON
# -----------------------
def _flatten(obj, prefix="", out=None):
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            _flatten(v, key, out)
    elif isinstance(obj, list):
        # For lists, create indexed keys and also try to aggregate scalar lists
        if all(not isinstance(i, (dict, list)) for i in obj):
            # list of scalars -> join as pipe
            out[prefix] = "|".join([str(i) for i in obj if i is not None])
        else:
            for idx, item in enumerate(obj):
                key = f"{prefix}.{idx}" if prefix else str(idx)
                _flatten(item, key, out)
    else:
        out[prefix] = obj
    return out

def json_to_records(file_bytes):
    try:
        text = file_bytes.read().decode("utf-8")
    except:
        file_bytes.seek(0)
        text = file_bytes.read().decode("latin-1")
    data = json.loads(text)
    # If root is list of items
    if isinstance(data, list):
        records = []
        for item in data:
            records.append(_flatten(item))
        return pd.DataFrame(records)
    # If dict with nested lists: try to find the largest list in the dict
    if isinstance(data, dict):
        # find lists
        candidate_lists = []
        def walk(d, path=""):
            if isinstance(d, dict):
                for k,v in d.items():
                    walk(v, f"{path}.{k}" if path else k)
            elif isinstance(d, list):
                candidate_lists.append((path, d))
        walk(data)
        if candidate_lists:
            # pick longest list
            path, lst = max(candidate_lists, key=lambda x: len(x[1]))
            records = []
            for item in lst:
                # try to merge surrounding root-level keys to each item
                merged = {}
                if isinstance(item, dict):
                    merged.update(item)
                records.append(_flatten(merged))
            return pd.DataFrame(records)
        # else single object -> single-row
        return pd.DataFrame([_flatten(data)])
    # fallback
    return pd.DataFrame()

# -----------------------
# Utilities: parse XML into list of records
# -----------------------
def xml_to_dict(elem):
    # convert element to dict recursively
    out = {}
    # attributes
    out.update(elem.attrib)
    # children
    children = list(elem)
    if children:
        child_groups = {}
        for ch in children:
            tag = ch.tag
            d = xml_to_dict(ch)
            child_groups.setdefault(tag, []).append(d)
        for tag, vals in child_groups.items():
            if len(vals) == 1:
                out[tag] = vals[0]
            else:
                out[tag] = vals
    # text
    text = (elem.text or "").strip()
    if text and not children and not elem.attrib:
        return text
    elif text:
        out["_text"] = text
    return out

def xml_to_records(file_bytes):
    try:
        text = file_bytes.read().decode("utf-8")
    except:
        file_bytes.seek(0)
        text = file_bytes.read().decode("latin-1")
    root = ET.fromstring(text)
    # detect repeating tags under root (candidate item nodes)
    tag_counts = {}
    for child in root:
        tag_counts[child.tag] = tag_counts.get(child.tag, 0) + 1
    # if there are repeating tags, pick the tag with most counts as item container
    repeating = [tag for tag, cnt in tag_counts.items() if cnt > 1]
    records = []
    if repeating:
        item_tag = max(repeating, key=lambda t: tag_counts[t])
        for item in root.findall(f'.//{item_tag}'):
            d = xml_to_dict(item)
            records.append(_flatten(d))
    else:
        # If no repeating tags, treat each direct child as record
        for item in list(root):
            d = xml_to_dict(item)
            records.append(_flatten(d))
        if not records:
            # fallback: single record from whole doc
            records = [_flatten(xml_to_dict(root))]
    return pd.DataFrame(records)

# -----------------------
# Validation helpers
# -----------------------
def parse_requirement(req_str):
    # Expect values like "Required", "Recommended", "Optional" (case-insensitive)
    if pd.isna(req_str):
        return "Optional"
    s = str(req_str).strip().lower()
    if "required" in s:
        return "Required"
    if "recommend" in s:
        return "Recommended"
    return "Optional"

def check_type(value_series, dtype_str):
    # dataframe series -> return (ok:boolean, details:str)
    if dtype_str is None or dtype_str == "" or pd.isna(dtype_str):
        return True, ""
    s = str(dtype_str).lower()
    non_null = value_series.dropna().astype(str)
    if non_null.empty:
        return True, ""
    if "int" in s and "float" not in s:
        coerced = pd.to_numeric(non_null, errors="coerce")
        if coerced.isna().any():
            return False, "Contains non-integer values"
        return True, ""
    if "float" in s or ("number" in s and "int" not in s):
        coerced = pd.to_numeric(non_null, errors="coerce")
        if coerced.isna().any():
            return False, "Contains non-numeric values"
        return True, ""
    if "bool" in s or "boolean" in s:
        ok = non_null.str.lower().isin(["true","false","0","1","yes","no"])
        if not ok.all():
            return False, "Contains non-boolean values"
        return True, ""
    if "date" in s or "datetime" in s:
        def try_parse(x):
            for fmt in ("%Y-%m-%d","%Y-%m-%dT%H:%M:%S","%d-%m-%Y","%m/%d/%Y","%Y/%m/%d"):
                try:
                    datetime.strptime(x, fmt)
                    return True
                except:
                    pass
            return False
        ok = non_null.apply(try_parse)
        if not ok.all():
            return False, "Contains values not parseable as dates"
        return True, ""
    if "url" in s:
        pattern = re.compile(r"^https?://")
        ok = non_null.apply(lambda x: bool(pattern.search(x)))
        if not ok.all():
            return False, "Contains values not starting with http(s)://"
        return True, ""
    # default: treat as string
    return True, ""

def check_supported_values(series, supported_str):
    if pd.isna(supported_str) or supported_str == "":
        return True, ""
    allowed = [s.strip().lower() for s in re.split(r"[|,;]", str(supported_str)) if s.strip()!=""]
    if not allowed:
        return True, ""
    non_null = series.dropna().astype(str).str.lower()
    if non_null.empty:
        return True, ""
    bad = ~non_null.isin(allowed)
    if bad.any():
        # show up to 5 examples
        examples = list(non_null[bad].unique()[:5])
        return False, f"Values outside allowed set. Examples: {examples}"
    return True, ""

def apply_validation_rules(series, rules_text):
    # rules_text may contain patterns like:
    # regex:^SKU-\d{4}$;max_length:50;min_length:2;unique:yes;required_if:other_field=xyz
    if pd.isna(rules_text) or rules_text == "":
        return True, ""
    rules = {}
    # split by semicolon
    parts = [p.strip() for p in str(rules_text).split(";") if p.strip()!=""]
    for p in parts:
        if ":" in p:
            k,v = p.split(":",1)
            rules[k.strip().lower()] = v.strip()
        else:
            rules[p.strip().lower()] = True
    details = []
    s = series.dropna().astype(str)
    if "max_length" in rules:
        try:
            mx = int(rules["max_length"])
            too_long = s.apply(len) > mx
            if too_long.any():
                details.append(f"{too_long.sum()} values exceed max_length {mx}")
        except:
            pass
    if "min_length" in rules:
        try:
            mn = int(rules["min_length"])
            too_short = s.apply(len) < mn
            if too_short.any():
                details.append(f"{too_short.sum()} values shorter than min_length {mn}")
        except:
            pass
    if "regex" in rules:
        try:
            pattern = re.compile(rules["regex"])
            bad = ~s.apply(lambda x: bool(pattern.search(x)))
            if bad.any():
                details.append(f"{bad.sum()} values do not match regex")
        except Exception as e:
            details.append(f"Invalid regex: {e}")
    if "unique" in rules and str(rules["unique"]).lower() in ("yes","true","1"):
        dup = s.duplicated().sum()
        if dup > 0:
            details.append(f"{dup} duplicate values found (should be unique)")
    # other rules may be custom; return details
    ok = len(details) == 0
    return ok, "; ".join(details)

def evaluate_dependencies(row_values_map, dependencies_text):
    # Very simple parser: expects rules like "if field=value then required" or "fieldA -> fieldB"
    # We'll support patterns: "field=value", "field==value", or "field -> required:other_field"
    if pd.isna(dependencies_text) or dependencies_text == "":
        return True, ""
    text = str(dependencies_text)
    # handle common simple pattern: "other_field=val => required"
    # We'll return a list of triggered dependency warnings/errors to the validator caller
    # For now just return True (no automatic failure) and an explanatory note if rule applies
    notes = []
    # try to split on semicolons
    parts = [p.strip() for p in re.split(r"[;|]", text) if p.strip()!=""]
    for p in parts:
        m = re.search(r"(\w[\w\.\-]*)\s*(=|==|:)\s*([^\s>]+)", p)
        if m:
            field, _, val = m.group(1), m.group(2), m.group(3)
            fv = row_values_map.get(field)
            if fv is not None and str(fv).lower() == val.lower():
                notes.append(f"Dependency triggered: {field} == {val}")
    return True, ("; ".join(notes)) if notes else ""

# -----------------------
# Main validation function
# -----------------------
def validate_against_spec(spec_df, feed_df):
    # normalize column labels
    spec_df = spec_df.copy()
    spec_df['Attribute'] = spec_df['Attribute'].astype(str).str.strip()
    spec_df['attribute_norm'] = spec_df['Attribute'].str.lower()
    feed_cols = list(feed_df.columns)
    feed_cols_norm = [c.lower() for c in feed_cols]
    report = []

    # For dependency evaluation we'll sample first row values (if available)
    sample_map = {}
    if not feed_df.empty:
        sample_row = feed_df.iloc[0].to_dict()
        sample_map = {k: (v if pd.notna(v) else None) for k,v in sample_row.items()}

    for idx, spec_row in spec_df.iterrows():
        attr = spec_row['Attribute']
        attr_norm = spec_row['attribute_norm']
        dtype = spec_row.get('Data Type', "")
        supported = spec_row.get('Supported Values', "")
        description = spec_row.get('Description', "")
        example = spec_row.get('Example', "")
        requirement = parse_requirement(spec_row.get('Requirement', ""))
        dependencies = spec_row.get('Dependencies', "")
        validation_rules = spec_row.get('Validation Rules', "")

        exists = attr_norm in feed_cols_norm
        status = "✅ Present" if exists else "❌ Missing"
        details = []

        # existence + requirement
        if not exists:
            if requirement == "Required":
                status = "❌ Missing (Required)"
                details.append("This attribute is required by spec but not present in feed.")
            elif requirement == "Recommended":
                status = "⚠️ Missing (Recommended)"
                details.append("Recommended by spec but not present.")
            else:
                status = "ℹ️ Missing (Optional)"
                details.append("Optional and not present.")
            # still add row
            report.append({
                "Attribute": attr,
                "Requirement": requirement,
                "Exists in Feed": "No",
                "Status": status,
                "Details": " ".join(details),
                "Description": description,
                "Example": example
            })
            continue

        # if exists: gather column data
        orig_col = feed_cols[feed_cols_norm.index(attr_norm)]
        col_series = feed_df[orig_col]

        # emptiness
        empty_pct = col_series.isna().mean() * 100
        if empty_pct > 0:
            details.append(f"{empty_pct:.1f}% values empty")

        # data type check
        ok_type, tdet = check_type(col_series, dtype)
        if not ok_type:
            details.append(tdet)
            status = "⚠️ Type/Format Issue"

        # supported values check
        ok_vals, vdet = check_supported_values(col_series, supported)
        if not ok_vals:
            details.append(vdet)
            status = "⚠️ Value Issue"

        # validation rules
        ok_rules, rdet = apply_validation_rules(col_series, validation_rules)
        if not ok_rules:
            details.append(rdet)
            status = "⚠️ Validation Rules Failed"

        # dependencies (best-effort)
        _, dnote = evaluate_dependencies(sample_map, dependencies)
        if dnote:
            details.append(f"Dependencies: {dnote}")

        # If nothing flagged, keep Present
        if not details and status.startswith("✅"):
            status = "✅ Present & Valid"

        report.append({
            "Attribute": attr,
            "Requirement": requirement,
            "Exists in Feed": "Yes",
            "Status": status,
            "Details": " | ".join(details),
            "Description": description,
            "Example": example
        })

    # detect extra fields
    spec_attrs_norm = set(spec_df['attribute_norm'].tolist())
    extras = []
    for c in feed_cols:
        if c.lower() not in spec_attrs_norm:
            extras.append(c)
    extras_note = ""
    if extras:
        extras_note = f"Found {len(extras)} extra/unrecognized fields: {extras[:10]}"
        # append a summary row
        report.append({
            "Attribute": "(Extra fields)",
            "Requirement": "",
            "Exists in Feed": "Yes",
            "Status": "⚠️ Extra fields",
            "Details": extras_note,
            "Description": "Fields present in feed but not defined in spec",
            "Example": ""
        })

    return pd.DataFrame(report)

# -----------------------
# UI
# -----------------------
st.title("🔎 Auto-Schema Feed Validator — JSON & XML")
st.write("Upload your spec CSV (must include the columns in your spec) and a user feed in JSON or XML. The validator will auto-detect schema, flatten items, and run strict checks for every attribute listed in the spec.")

with st.sidebar:
    st.header("What to upload")
    st.write("- Your spec CSV (the same file you uploaded).")
    st.write("- A user feed in JSON OR XML (single file).")
    st.write("")
    st.markdown("**Spec columns detected in your file:**")
    st.caption("Attribute, Data Type, Supported Values, Description, Example, Requirement, Dependencies, Validation Rules")

spec_file = st.file_uploader("Upload Spec CSV", type=["csv"])
feed_file = st.file_uploader("Upload User Feed (JSON or XML)", type=["json","xml"])

if spec_file:
    try:
        spec_df = load_spec_df(spec_file)
        st.success("Spec loaded. Columns: " + ", ".join(spec_df.columns))
    except Exception as e:
        st.error(f"Failed to load spec CSV: {e}")
        spec_df = None
else:
    spec_df = None

if spec_df is not None and feed_file is not None:
    # parse feed
    try:
        feed_file.seek(0)
        if feed_file.name.lower().endswith(".json"):
            feed_df = json_to_records(feed_file)
        else:
            feed_file.seek(0)
            feed_df = xml_to_records(feed_file)
    except Exception as e:
        st.error(f"Failed to parse feed: {e}")
        feed_df = pd.DataFrame()

    if feed_df is None or feed_df.empty:
        st.warning("Parsed feed has no rows or couldn't be parsed into records. Check feed structure.")
    else:
        st.write("### Parsed feed preview (first 10 rows, flattened)")
        st.dataframe(feed_df.head(10), use_container_width=True)

        with st.spinner("Running validation..."):
            report_df = validate_against_spec(spec_df, feed_df)

        st.write("### Validation report")
        st.dataframe(report_df, use_container_width=True)

        csv_out = report_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download validation report (CSV)", data=csv_out, file_name="validation_report.csv", mime="text/csv")

        # Also offer a "suggested fixes" CSV skeleton: for missing required attributes, produce blank columns
        missing_required = report_df[report_df['Status'].str.contains("Missing") & (report_df['Requirement']=="Required")]
        if not missing_required.empty:
            st.warning(f"{len(missing_required)} required attributes missing.")
            if st.button("Generate suggested-corrected-feed skeleton (CSV)"):
                # create skeleton with required columns + present columns
                required_cols = missing_required['Attribute'].tolist()
                present_cols = [c for c in feed_df.columns]
                skeleton_cols = present_cols + required_cols
                skeleton_df = pd.DataFrame(columns=skeleton_cols)
                buf = skeleton_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download skeleton CSV", data=buf, file_name="skeleton_missing_required.csv", mime="text/csv")
