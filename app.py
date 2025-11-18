import streamlit as st
import pandas as pd
import json
import xml.etree.ElementTree as ET
from io import StringIO, BytesIO
import re
from datetime import datetime

st.set_page_config(page_title="Auto Schema Feed Validator — Embedded Spec", layout="wide")

# -------------------------
# EMBEDDED SPEC CSV (TAB-SEP)
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

# -------------------------
# Read spec into DataFrame
# -------------------------
@st.cache_data
def load_spec_df_from_string(spec_text: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(spec_text), sep="\t", engine="python")
    df.columns = [c.strip() for c in df.columns]
    return df

spec_df = load_spec_df_from_string(SPEC_CSV)

# -------------------------
# Flatten helpers (JSON)
# -------------------------
def _flatten_json(obj, prefix="", out=None):
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            _flatten_json(v, key, out)
    elif isinstance(obj, list):
        # if list of scalars, join; else index
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
        # find largest list anywhere
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
        # single object
        return pd.DataFrame([_flatten_json(data)])
    return pd.DataFrame()

# -------------------------
# XML parsing -> records
# -------------------------
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
    # default string
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
        if p.startswith("max"):
            m = re.search(r"max\s*\D*(\d+)", p)
            if m:
                mx = int(m.group(1))
                too_long = s.apply(len) > mx
                if too_long.any():
                    details.append(f"{too_long.sum()} values exceed max length {mx}")
        elif p.startswith("min"):
            m = re.search(r"min\s*\D*(\d+)", p)
            if m:
                mn = int(m.group(1))
                too_short = s.apply(len) < mn
                if too_short.any():
                    details.append(f"{too_short.sum()} values shorter than min length {mn}")
        elif p.startswith("regex:"):
            pattern = p.split("regex:",1)[1]
            try:
                pat = re.compile(pattern)
                bad = ~s.apply(lambda x: bool(pat.search(x)))
                if bad.any():
                    details.append(f"{bad.sum()} values do not match regex")
            except Exception as e:
                details.append(f"Invalid regex: {e}")
        elif p.lower().startswith("unique"):
            if "yes" in p.lower() or "true" in p.lower():
                dup = s.duplicated().sum()
                if dup>0:
                    details.append(f"{dup} duplicate values (should be unique)")
    ok = len(details) == 0
    return ok, "; ".join(details)

# -------------------------
# Main validation
# -------------------------
def validate_feed(spec_df: pd.DataFrame, feed_df: pd.DataFrame):
    spec_df = spec_df.copy()
    spec_df['Attribute_norm'] = spec_df['Attribute'].astype(str).str.strip().str.lower()
    feed_cols = list(feed_df.columns)
    feed_cols_norm = [c.lower() for c in feed_cols]

    report_rows = []
    # sample map for dependency checking (first row)
    sample_map = {}
    if not feed_df.empty:
        first = feed_df.iloc[0].to_dict()
        sample_map = {k: (v if pd.notna(v) else None) for k,v in first.items()}

    for _, srow in spec_df.iterrows():
        attr = srow['Attribute']
        attr_norm = srow['Attribute_norm']
        dtype = srow.get('Data Type', "")
        supported = srow.get('Supported Values', "")
        requirement = parse_requirement(srow.get('Requirement', ""))
        validation_rules = srow.get('Validation Rules', "")
        dependencies = srow.get('Dependencies', "")

        exists = attr_norm in feed_cols_norm
        if not exists:
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

        # present -> run checks (against all rows)
        orig_col = feed_cols[feed_cols_norm.index(attr_norm)]
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

        ok_rules, rdet = apply_validation_rules(col, validation_rules)
        if not ok_rules:
            details.append(rdet)

        # dependencies best-effort (informational)
        dep_notes = []
        if not pd.isna(dependencies) and str(dependencies).strip() != "":
            # simple check: if "enabled_checkout" mentioned and sample_map shows missing or false, note it
            if "enabled_checkout" in str(dependencies).lower() and sample_map.get("enable_checkout") not in (True,"true","TRUE","True","1","yes"):
                dep_notes.append("Dependency note: enable_checkout not true in sample row")
        if dep_notes:
            details += dep_notes

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

    # extras
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
st.title("🔎 Auto Schema Feed Validator — Embedded Spec")
st.write("Spec is embedded in the app. Upload a JSON or XML product feed. The entire feed will be flattened and validated against the embedded spec.")

st.sidebar.header("About")
st.sidebar.write("Spec columns: Attribute, Data Type, Supported Values, Description, Example, Requirement, Dependencies, Validation Rules")
st.sidebar.write("Note: Validation runs on the full dataset (all rows).")

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
            st.success(f"Parsed {len(feed_df)} record(s). Running full validation (all rows)...")
            with st.spinner("Validating (this checks all rows)..."):
                report = validate_feed(spec_df, feed_df)

            st.write("### Validation report (per-attribute)")
            st.dataframe(report, use_container_width=True)

            csv_bytes = report.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download validation report (CSV)", csv_bytes, file_name="validation_report.csv", mime="text/csv")

            # Offer skeleton for missing required attributes
            missing_required = report[report['Status'].str.contains("Missing") & (report['Requirement']=="Required")]
            if not missing_required.empty:
                st.warning(f"{len(missing_required)} required attributes missing from feed.")
                if st.button("Generate skeleton CSV with missing required columns"):
                    required_cols = missing_required['Attribute'].tolist()
                    skeleton_cols = list(feed_df.columns) + required_cols
                    skeleton_df = pd.DataFrame(columns=skeleton_cols)
                    st.download_button("⬇️ Download skeleton CSV", skeleton_df.to_csv(index=False).encode("utf-8"), "skeleton.csv", "text/csv")
    except Exception as e:
        st.error(f"Failed to parse/validate feed: {e}")
