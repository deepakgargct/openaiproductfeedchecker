# app.py
import streamlit as st
import pandas as pd
import json
import xml.etree.ElementTree as ET
from io import StringIO, BytesIO
import re
from datetime import datetime

st.set_page_config(page_title="Auto Schema Feed Validator — Product-Level", layout="wide")

# -------------------------
# EMBEDDED SPEC (TAB-SEP)
# -------------------------
SPEC_CSV = """Attribute\tData Type\tSupported Values\tDescription\tExample\tRequirement\tDependencies\tValidation Rules
enable_search\tEnum\ttrue, false\tControls whether the product can be surfaced in ChatGPT search results.\tTRUE\tRequired\t—\tLower-case string
enable_checkout\tEnum\ttrue, false\t"Allows direct purchase inside ChatGPT.
enable_search must be true in order for enable_checkout to be enabled for the product."\tTRUE\tRequired\t—\tLower-case string
id\tString (alphanumeric)\t—\tMerchant product ID (unique)\tSKU12345\tRequired\t—\tMax 100 chars; must remain stable over time
gtin\tString (numeric)\tGTIN, UPC, ISBN\tUniversal product identifier\t1.23457E+11\tRecommended\t—\t8–14 digits; no dashes or spaces
mpn\tString (alphanumeric)\t—\tManufacturer part number\tGPT5\tRequired if gtin missing\tRequired if gtin is absent\tMax 70 chars
title\tString (UTF-8 text)\t—\tProduct title\tMen's Trail Running Shoes Black\tRequired\t—\tMax 150 chars; avoid all-caps
description\tString (UTF-8 text)\t—\tFull product description\tWaterproof trail shoe with cushioned sole…\tRequired\t—\tMax 5,000 chars; plain text only
link\tURL\tRFC 1738\tProduct detail page URL\thttps://example.com/product/SKU12345\tRequired\t—\tMust resolve with HTTP 200; HTTPS preferred
condition\tEnum\tnew, refurbished, used\tCondition of product\tnew\tRequired if product condition differs from new\t—\tLower-case string
product_category\tString\tCategory taxonomy\tCategory path\tApparel & Accessories > Shoes\tRequired\t—\tUse “>” separator
brand\tString\t—\tProduct brand\tOpenAI\tRequired for all excluding movies, books, and musical recording brands\t—\tMax 70 chars
material\tString\t—\tPrimary material(s)\tLeather\tRequired\t—\tMax 100 chars
dimensions\tString\tLxWxH unit\tOverall dimensions\t12x8x5 in\tOptional\t—\tUnits required if provided
length\tNumber + unit\t—\tIndividual dimension\t10 mm\tOptional\tProvide all three if using individual fields\tUnits required
width\tNumber + unit\t—\tIndividual dimension\t10 mm\tOptional\tProvide all three if using individual fields\tUnits required
height\tNumber + unit\t—\tIndividual dimension\t10 mm\tOptional\tProvide all three if using individual fields\tUnits required
weight\tNumber + unit\t—\tProduct weight\t1.5 lb\tRequired\t—\tPositive number with unit
age_group\tEnum\tnewborn, infant, toddler, kids, adult\tTarget demographic\tadult\tOptional\t—\tLower-case string
image_link\tURL\tRFC 1738\tMain product image URL\thttps://example.com/image1.jpg\tRequired\t—\tJPEG/PNG; HTTPS preferred
additional_image_link\tURL array\tRFC 1738\tExtra images\thttps://example.com/image2.jpg,…\tOptional\t—\tComma-separated or array
video_link\tURL\tRFC 1738\tProduct video\thttps://youtu.be/12345\tOptional\t—\tMust be publicly accessible
model_3d_link\tURL\tRFC 1738\t3D model\thttps://example.com/model.glb\tOptional\t—\tGLB/GLTF preferred
price\tNumber + currency\tISO 4217\tRegular price\t79.99 USD\tRequired\t—\tMust include currency code
sale_price\tNumber + currency\tISO 4217\tDiscounted price\t59.99 USD\tOptional\t—\tMust be ≤ price
sale_price_effective_date\tDate range\tISO 8601\tSale window\t2025-07-01 / 2025-07-15\tOptional\tRequired if sale_price provided\tStart must precede end
unit_pricing_measure / base_measure\tNumber + unit\t—\tUnit price & base measure\t16 oz / 1 oz\tOptional\t—\tBoth fields required together
pricing_trend\tString\t—\tLowest price in N months\tLowest price in 6 months\tOptional\t—\tMax 80 chars
availability\tEnum\tin_stock, out_of_stock, preorder\tProduct availability\tin_stock\tRequired\t—\tLower-case string
availability_date\tDate\tISO 8601\tAvailability date if preorder\t01-12-2025\tRequired if availability=preorder\t—\tMust be future date
inventory_quantity\tInteger\t—\tStock count\t25\tRequired\t—\tNon-negative integer
expiration_date\tDate\tISO 8601\tRemove product after date\t01-12-2025\tOptional\t—\tMust be future date
pickup_method\tEnum\tin_store, reserve, not_supported\tPickup options\tin_store\tOptional\t—\tLower-case string
pickup_sla\tNumber + duration\t—\tPickup SLA\t1 day\tOptional\tRequires pickup_method\tPositive integer + unit
item_group_id\tString\t—\tVariant group ID\tSHOE123GROUP\tRequired if variants exist\t—\tMax 70 chars
item_group_title\tString (UTF-8 text)\t—\tGroup product title\tMen's Trail Running Shoes\tOptional\t—\tMax 150 chars; avoid all-caps
color\tString\t—\tVariant color\tBlue\tRecommended (apparel)\t—\tMax 40 chars
size\tString\t—\tVariant size\t10\tRecommended (apparel)\t—\tMax 20 chars
size_system\tCountry code\tISO 3166\tSize system\tUS\tRecommended (apparel)\t—\t2-letter country code
gender\tEnum\tmale, female, unisex\tGender target\tmale\tRecommended (apparel)\t—\tLower-case string
offer_id\tString\t—\tOffer ID (SKU+seller+price)\tSKU12345-Blue-79.99\tRecommended\t—\tUnique within feed
Custom_variant1_category\tString\t—\tCustom variant dimension 1\tSize_Type\tOptional\t—\t—
Custom_variant1_option\tString\t—\tCustom variant 1 option\tPetite / Tall / Maternity\tOptional\t—\t—
Custom_variant2_category\tString\t—\tCustom variant dimension 2\tWood_Type\tOptional\t—\t—
Custom_variant2_option\tString\t—\tCustom variant 2 option\tOak / Mahogany / Walnut\tOptional\t—\t—
Custom_variant3_category\tString\t—\tCustom variant dimension 3\tCap_Type\tOptional\t—\t—
Custom_variant3_option\tString\t—\tCustom variant 3 option\tSnapback / Fitted\tOptional\t—\t—
shipping\tString\tcountry:region:service_class:price\tShipping method/cost/region\tUS:CA:Overnight:16.00 USD\tRequired where applicable\t—\tMultiple entries allowed; use colon separators
delivery_estimate\tDate\tISO 8601\tEstimated arrival date\t12-08-2025\tOptional\t—\tMust be future date
seller_name\tString\t—\tSeller name\tExample Store\tRequired / Display\t—\tMax 70 chars
seller_url\tURL\tRFC 1738\tSeller page\thttps://example.com/store\tRequired\t—\tHTTPS preferred
seller_privacy_policy\tURL\tRFC 1738\tSeller-specific policies\thttps://example.com/privacy\tRequired, if enabled_checkout is true\t—\tHTTPS preferred
seller_tos\tURL\tRFC 1738\tSeller-specific terms of service\thttps://example.com/terms\tRequired, if enabled_checkout is true\t—\tHTTPS preferred
return_policy\tURL\tRFC 1738\tReturn policy URL\thttps://example.com/returns\tRequired\t—\tHTTPS preferred
return_window\tInteger\tDays\tDays allowed for return\t30\tRequired\t—\tPositive integer
popularity_score\tNumber\t—\tPopularity indicator\t4.7\tRecommended\t—\t0–5 scale or merchant-defined
return_rate\tNumber\tPercentage\tReturn rate\t2%\tRecommended\t—\t0–100%
warning / warning_url\tString / URL\t—\tProduct disclaimers\tContains lithium battery, or CA Prop 65 warning\tRecommended for Checkout\t—\tIf URL, must resolve HTTP 200
age_restriction\tNumber\t—\tMinimum purchase age\t21\tRecommended\t—\tPositive integer
product_review_count\tInteger\t—\tNumber of product reviews\t254\tRecommended\t—\tNon-negative
product_review_rating\tNumber\t—\tAverage review score\t4.6\tRecommended\t—\t0–5 scale
store_review_count\tInteger\t—\tNumber of brand/store reviews\t2000\tOptional\t—\tNon-negative
store_review_rating\tNumber\t—\tAverage store rating\t4.8\tOptional\t—\t0–5 scale
q_and_a\tString\t—\tFAQ content\tQ: Is this waterproof? A: Yes\tRecommended\t—\tPlain text
raw_review_data\tString\t—\tRaw review payload\t—\tRecommended\t—\tMay include JSON blob
related_product_id\tString\t—\tAssociated product IDs\tSKU67890\tRecommended\t—\tComma-separated list allowed
relationship_type\tEnum\tpart_of_set, required_part, often_bought_with, substitute, different_brand, accessory\tRelationship type\tpart_of_set\tRecommended\t—\tLower-case string
geo_price\tNumber + currency\tRegion-specific price\tPrice by region\t79.99 USD (California)\tRecommended\t—\tMust include ISO 4217 currency
geo_availability\tString\tRegion-specific availability\tAvailability per region\tin_stock (Texas), out_of_stock (New York)\tRecommended\t—\tRegions must be valid ISO 3166 codes
"""

@st.cache_data
def load_spec_df_from_string(spec_text: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(spec_text), sep="\t", engine="python")
    df.columns = [c.strip() for c in df.columns]
    return df

spec_df = load_spec_df_from_string(SPEC_CSV)

# -------------------------
# Flatten helpers
# -------------------------
def _flatten_json(obj, prefix="", out=None):
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            _flatten_json(v, key, out)
    elif isinstance(obj, list):
        # if list of scalars, join; else index each element
        if all(not isinstance(i, (dict, list)) for i in obj):
            out[prefix] = "|".join([str(i) for i in obj if i is not None])
        else:
            for idx, item in enumerate(obj):
                key = f"{prefix}.{idx}" if prefix else str(idx)
                _flatten_json(item, key, out)
    else:
        out[prefix] = obj
    return out

def json_to_records(file_bytes: BytesIO) -> pd.DataFrame:
    file_bytes.seek(0)
    data_text = file_bytes.read().decode("utf-8", errors="ignore")
    data = json.loads(data_text)
    records = []
    if isinstance(data, list):
        for item in data:
            records.append(_flatten_json(item))
        return pd.DataFrame(records)
    if isinstance(data, dict):
        # locate largest list anywhere and treat it as records
        candidate_lists = []
        def walk(d):
            if isinstance(d, dict):
                for k,v in d.items():
                    walk(v)
            elif isinstance(d, list):
                candidate_lists.append(d)
        walk(data)
        if candidate_lists:
            largest = max(candidate_lists, key=len)
            for item in largest:
                merged = {}
                if isinstance(item, dict):
                    merged.update(item)
                records.append(_flatten_json(merged))
            return pd.DataFrame(records)
        return pd.DataFrame([_flatten_json(data)])
    return pd.DataFrame()

def xml_to_dict(elem):
    out = {}
    out.update(elem.attrib)
    children = list(elem)
    if children:
        cg = {}
        for ch in children:
            tag = ch.tag
            d = xml_to_dict(ch)
            cg.setdefault(tag, []).append(d)
        for tag, vals in cg.items():
            if len(vals) == 1:
                out[tag] = vals[0]
            else:
                out[tag] = vals
    text = (elem.text or "").strip()
    if text and not children and not elem.attrib:
        return text
    elif text:
        out["_text"] = text
    return out

def xml_to_records(file_bytes: BytesIO) -> pd.DataFrame:
    file_bytes.seek(0)
    text = file_bytes.read().decode("utf-8", errors="ignore")
    root = ET.fromstring(text)
    tag_counts = {}
    for child in root:
        tag_counts[child.tag] = tag_counts.get(child.tag, 0) + 1
    repeating = [t for t,c in tag_counts.items() if c > 1]
    records = []
    if repeating:
        container = max(repeating, key=lambda t: tag_counts[t])
        for item in root.findall(f'.//{container}'):
            d = xml_to_dict(item)
            records.append(_flatten_json(d))
    else:
        for item in list(root):
            d = xml_to_dict(item)
            records.append(_flatten_json(d))
        if not records:
            records = [_flatten_json(xml_to_dict(root))]
    return pd.DataFrame(records)

# -------------------------
# Validation helpers
# -------------------------
def parse_requirement(req):
    if pd.isna(req):
        return "Optional"
    s = str(req).strip().lower()
    if "required" in s:
        return "Required"
    if "recommend" in s:
        return "Recommended"
    return "Optional"

def check_type(series: pd.Series, dtype_str: str):
    if dtype_str is None or pd.isna(dtype_str) or dtype_str == "":
        return True, ""
    s = str(dtype_str).lower()
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return True, ""
    if "int" in s and "float" not in s:
        coerced = pd.to_numeric(non_null, errors="coerce")
        if coerced.isna().any():
            return False, "Non-integer values found"
        return True, ""
    if "number" in s or "float" in s:
        coerced = pd.to_numeric(non_null.str.replace(r"[^\d\.\-]", "", regex=True), errors="coerce")
        if coerced.isna().any():
            return False, "Non-numeric values found"
        return True, ""
    if "date" in s or "iso" in s:
        def try_parse(x):
            for fmt in ("%Y-%m-%d","%d-%m-%Y","%m/%d/%Y","%Y/%m/%d","%Y-%m-%dT%H:%M:%S"):
                try:
                    datetime.strptime(x, fmt)
                    return True
                except:
                    pass
            return False
        ok = non_null.apply(try_parse)
        if not ok.all():
            return False, "Values not parseable as dates"
        return True, ""
    if "url" in s:
        pattern = re.compile(r"^https?://", re.I)
        ok = non_null.apply(lambda x: bool(pattern.search(x)))
        if not ok.all():
            return False, "Some values do not start with http(s)://"
        return True, ""
    return True, ""

def check_supported_values(series: pd.Series, supported_str: str):
    if pd.isna(supported_str) or supported_str == "":
        return True, ""
    allowed = [a.strip().lower() for a in re.split(r"[,|;]", str(supported_str)) if a.strip()!=""]
    if not allowed:
        return True, ""
    non_null = series.dropna().astype(str).str.lower()
    if non_null.empty:
        return True, ""
    bad_mask = ~non_null.isin(allowed)
    if bad_mask.any():
        examples = list(non_null[bad_mask].unique()[:5])
        return False, f"Values outside allowed set. Examples: {examples}"
    return True, ""

def apply_validation_rules(series: pd.Series, rules_text: str):
    if pd.isna(rules_text) or rules_text == "":
        return True, ""
    parts = [p.strip() for p in str(rules_text).split(";") if p.strip()!=""]
    details = []
    s = series.dropna().astype(str)
    for p in parts:
        low = p.lower()
        if low.startswith("max"):
            m = re.search(r"max\s*\D*(\d+)", p)
            if m:
                mx = int(m.group(1))
                too_long = s.apply(len) > mx
                if too_long.any():
                    details.append(f"{too_long.sum()} values exceed max length {mx}")
        elif low.startswith("min"):
            m = re.search(r"min\s*\D*(\d+)", p)
            if m:
                mn = int(m.group(1))
                too_short = s.apply(len) < mn
                if too_short.any():
                    details.append(f"{too_short.sum()} values shorter than min length {mn}")
        elif low.startswith("regex:"):
            pattern = p.split("regex:",1)[1]
            try:
                pat = re.compile(pattern)
                bad = ~s.apply(lambda x: bool(pat.search(x)))
                if bad.any():
                    details.append(f"{bad.sum()} values do not match regex")
            except Exception as e:
                details.append(f"Invalid regex: {e}")
        elif "unique" in low:
            if "yes" in low or "true" in low:
                dup = s.duplicated().sum()
                if dup>0:
                    details.append(f"{dup} duplicate values (should be unique)")
    ok = len(details) == 0
    return ok, "; ".join(details)

# -------------------------
# Per-product validator
# -------------------------
def validate_product_record(record: dict, spec_df: pd.DataFrame):
    """
    record: flattened dict for one product
    returns: dict {missing:[], empty:[], type_issues:[], value_issues:[], validation_issues:[], extras:[]}
    """
    issues = {
        "missing": [],
        "empty": [],
        "type_issues": [],
        "value_issues": [],
        "validation_issues": [],
        "extras": []
    }
    rec_keys_norm = {k.lower(): k for k in record.keys()}

    spec_attrs = spec_df['Attribute'].tolist()
    spec_norm = [a.lower() for a in spec_attrs]

    # check for extras
    for k in record.keys():
        if k.lower() not in spec_norm:
            issues["extras"].append(k)

    # check each spec attribute
    for _, srow in spec_df.iterrows():
        attr = srow['Attribute']
        a_norm = attr.lower()
        requirement = parse_requirement(srow.get('Requirement', ""))
        dtype = srow.get('Data Type', "")
        supported = srow.get('Supported Values', "")
        vrules = srow.get('Validation Rules', "")

        if a_norm not in rec_keys_norm:
            if requirement == "Required":
                issues["missing"].append(attr)
            elif requirement == "Recommended":
                # recommended missing flagged as warning in missing for product-level
                issues["missing"].append(attr + " (recommended)")
            continue

        val = record.get(rec_keys_norm[a_norm])
        # check empty
        if val is None or (isinstance(val, str) and str(val).strip() == ""):
            # consider required/important emptiness
            if requirement == "Required":
                issues["empty"].append(attr)
            continue

        # simple single-value checks
        series = pd.Series([val])
        ok_type, tdet = check_type(series, dtype)
        if not ok_type:
            issues["type_issues"].append(f"{attr}: {tdet}")

        ok_vals, vdet = check_supported_values(series, supported)
        if not ok_vals:
            issues["value_issues"].append(f"{attr}: {vdet}")

        ok_rules, rdet = apply_validation_rules(series, vrules)
        if not ok_rules:
            issues["validation_issues"].append(f"{attr}: {rdet}")

    return issues

# -------------------------
# Field coverage summary
# -------------------------
def coverage_summary(feed_df: pd.DataFrame, spec_df: pd.DataFrame):
    rows = []
    for _, srow in spec_df.iterrows():
        attr = srow['Attribute']
        a_norm = attr.lower()
        present = False
        filled_pct = 0.0
        example = ""
        for c in feed_df.columns:
            if c.lower() == a_norm:
                present = True
                filled_pct = 100.0 - feed_df[c].isna().mean() * 100.0
                non_nulls = feed_df[c].dropna().astype(str)
                example = non_nulls.iloc[0] if not non_nulls.empty else ""
                break
        rows.append({
            "Attribute": attr,
            "Present": present,
            "% Filled": f"{filled_pct:.1f}%" if present else "0.0%",
            "Example Value": example
        })
    # also list extras
    spec_norm_set = set([a.lower() for a in spec_df['Attribute'].tolist()])
    extras = [c for c in feed_df.columns if c.lower() not in spec_norm_set]
    if extras:
        rows.append({
            "Attribute": "(Extra fields)",
            "Present": True,
            "% Filled": "",
            "Example Value": ", ".join(extras[:20])
        })
    return pd.DataFrame(rows)

# -------------------------
# Main validate_feed (attribute-level)
# -------------------------
def validate_feed_attribute_level(spec_df: pd.DataFrame, feed_df: pd.DataFrame):
    spec_df = spec_df.copy()
    spec_df['Attribute_norm'] = spec_df['Attribute'].astype(str).str.strip().str.lower()
    feed_cols = list(feed_df.columns)
    feed_cols_norm = [c.lower() for c in feed_cols]
    report_rows = []

    for _, srow in spec_df.iterrows():
        attr = srow['Attribute']
        a_norm = srow['Attribute_norm']
        dtype = srow.get('Data Type', "")
        supported = srow.get('Supported Values', "")
        requirement = parse_requirement(srow.get('Requirement', ""))
        vrules = srow.get('Validation Rules', "")

        if a_norm not in feed_cols_norm:
            if requirement == "Required":
                status = "❌ Missing (Required)"
            elif requirement == "Recommended":
                status = "⚠️ Missing (Recommended)"
            else:
                status = "ℹ️ Missing (Optional)"
            report_rows.append({
                "Attribute": attr,
                "Requirement": requirement,
                "Exists in Feed": "No",
                "Status": status,
                "Details": "Not present in feed",
                "Description": srow.get("Description",""),
                "Example": srow.get("Example","")
            })
            continue

        orig_col = feed_cols[feed_cols_norm.index(a_norm)]
        col = feed_df[orig_col]
        details = []
        empty_pct = col.isna().mean() * 100
        if empty_pct > 0:
            details.append(f"{empty_pct:.1f}% empty values")

        ok_type, tdet = check_type(col, dtype)
        if not ok_type:
            details.append(tdet)

        ok_vals, vdet = check_supported_values(col, supported)
        if not ok_vals:
            details.append(vdet)

        ok_rules, rdet = apply_validation_rules(col, vrules)
        if not ok_rules:
            details.append(rdet)

        status = "✅ Present & Valid" if not details else "⚠️ Issues"
        report_rows.append({
            "Attribute": attr,
            "Requirement": requirement,
            "Exists in Feed": "Yes",
            "Status": status,
            "Details": " | ".join(details) if details else "",
            "Description": srow.get("Description",""),
            "Example": srow.get("Example","")
        })

    # extras summary
    spec_set = set(spec_df['Attribute_norm'].tolist())
    extras = [c for c in feed_cols if c.lower() not in spec_set]
    if extras:
        report_rows.append({
            "Attribute": "(Extra fields)",
            "Requirement": "",
            "Exists in Feed": "Yes",
            "Status": f"⚠️ Extra fields ({len(extras)})",
            "Details": f"Extra / unrecognized fields: {extras[:20]}",
            "Description": "Fields present in feed but not specified",
            "Example": ""
        })

    return pd.DataFrame(report_rows)

# -------------------------
# UI
# -------------------------
st.title("🔎 Auto Schema Feed Validator — Product-Level (id used as identifier)")
st.write("Spec is embedded in the app. Upload JSON or XML product feed. The entire feed is validated and each product has an expandable panel showing issues & raw fields.")

st.sidebar.header("Controls")
show_limit = st.sidebar.number_input("Max products to expand in UI (set to avoid UI overload)", min_value=10, max_value=2000, value=200, step=10)
show_all_products = st.sidebar.checkbox("Allow expanding beyond limit (may be slow for very large feeds)", value=False)
st.sidebar.write("You can still download full product-level CSVs for deeper analysis.")

uploaded = st.file_uploader("Upload product feed (JSON or XML)", type=["json","xml"])
if uploaded is None:
    st.info("Upload a JSON or XML feed to start validation.")
else:
    try:
        uploaded.seek(0)
        if uploaded.name.lower().endswith(".json"):
            feed_df = json_to_records(uploaded)
        else:
            feed_df = xml_to_records(uploaded)

        if feed_df is None or feed_df.empty:
            st.error("Unable to parse feed into records. Check structure.")
        else:
            # normalize column names to be strings
            feed_df.columns = [str(c) for c in feed_df.columns]

            st.success(f"Parsed {len(feed_df)} record(s). Running validation on full dataset...")
            # attribute-level report
            with st.spinner("Running attribute-level validation..."):
                attr_report = validate_feed_attribute_level(spec_df, feed_df)

            st.markdown("## 1) Field coverage summary")
            cov = coverage_summary(feed_df, spec_df)
            st.dataframe(cov, use_container_width=True)
            st.download_button("⬇️ Download field coverage (CSV)", cov.to_csv(index=False).encode("utf-8"), "field_coverage.csv", "text/csv")

            st.markdown("## 2) Attribute-level validation report")
            st.dataframe(attr_report, use_container_width=True)
            st.download_button("⬇️ Download attribute-level report (CSV)", attr_report.to_csv(index=False).encode("utf-8"), "attribute_report.csv", "text/csv")

            # product-level checks (full)
            st.markdown("## 3) Product-level issues (expandable per-product)")
            product_issues_rows = []
            max_show = len(feed_df) if show_all_products else min(len(feed_df), int(show_limit))

            for idx, row in feed_df.iterrows():
                flattened_record = {k: row[k] for k in feed_df.columns}
                issues = validate_product_record(flattened_record, spec_df)

                # create summary string
                summary_parts = []
                if issues["missing"]:
                    summary_parts.append(f"Missing: {', '.join(issues['missing'][:8])}" + ("..." if len(issues['missing'])>8 else ""))
                if issues["empty"]:
                    summary_parts.append(f"Empty: {', '.join(issues['empty'][:8])}" + ("..." if len(issues['empty'])>8 else ""))
                if issues["type_issues"]:
                    summary_parts.append(f"Type: {', '.join(issues['type_issues'][:6])}" + ("..." if len(issues['type_issues'])>6 else ""))
                if issues["value_issues"]:
                    summary_parts.append(f"Value: {', '.join(issues['value_issues'][:6])}" + ("..." if len(issues['value_issues'])>6 else ""))
                if issues["validation_issues"]:
                    summary_parts.append(f"Rule: {', '.join(issues['validation_issues'][:6])}" + ("..." if len(issues['validation_issues'])>6 else ""))
                if issues["extras"]:
                    summary_parts.append(f"Extras: {', '.join(issues['extras'][:6])}" + ("..." if len(issues['extras'])>6 else ""))

                summary = " | ".join(summary_parts) if summary_parts else "No issues"

                # product identifier
                prod_id = flattened_record.get('id') or flattened_record.get('ID') or flattened_record.get('Id') or ""
                header = f"Product #{idx}"
                if prod_id:
                    header += f" — id: {prod_id}"

                # expandable UI only up to max_show (avoid rendering thousands)
                if idx < max_show:
                    with st.expander(f"{header} — {summary}", expanded=False):
                        st.write("**Summary:**")
                        st.write(summary)
                        st.write("**Issues detail:**")
                        st.json(issues)
                        st.write("**Raw fields for this product (flattened)**:")
                        st.dataframe(pd.DataFrame([flattened_record]).T.rename(columns={0:"value"}), use_container_width=True)
                # always collect product issues into CSV
                product_issues_rows.append({
                    "record_index": idx,
                    "id": prod_id,
                    "summary": summary,
                    "missing": "|".join(issues["missing"]),
                    "empty": "|".join(issues["empty"]),
                    "type_issues": "|".join(issues["type_issues"]),
                    "value_issues": "|".join(issues["value_issues"]),
                    "validation_issues": "|".join(issues["validation_issues"]),
                    "extras": "|".join(issues["extras"])
                })

            product_issues_df = pd.DataFrame(product_issues_rows)
            st.markdown("### Product-level issues summary")
            st.dataframe(product_issues_df.head(200), use_container_width=True)
            st.download_button("⬇️ Download full product-level issues CSV", product_issues_df.to_csv(index=False).encode("utf-8"), "product_issues.csv", "text/csv")

            st.info("If your feed is large, use the CSV downloads for complete analysis. You can increase 'Max products to expand' in the sidebar if you need to inspect more items in the UI.")
    except Exception as e:
        st.error(f"Failed to parse/validate feed: {e}")
