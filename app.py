# PART 1 of 4: Imports, Embedded Spec, Flattening, and Fuzzy Header Mapping
import streamlit as st
import pandas as pd
import json
import xml.etree.ElementTree as ET
from io import StringIO, BytesIO
import re
from datetime import datetime
import os

st.set_page_config(page_title="Auto Schema Feed Validator — with Fuzzy Mapping", layout="wide")

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

# -------------------------
# Load spec into DataFrame
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
# FUZZY HEADER MAPPER
# -------------------------
def canonicalize(col: str) -> str:
    """
    Normalize a column name to a fuzzy comparable key.
    Removes punctuation, underscores, numbers, and common suffixes/prefixes.
    """
    col = str(col).lower().strip()
    # replace non-alphanumerics with nothing
    col = re.sub(r"[^a-z0-9]+", "", col)
    # remove common prefixes
    col = re.sub(r"^(product|productid|productid|item|prod|p|merchant|seller|offer|offerid|offerid|sku|sku_code|skuid)","", col)
    # unify common suffixes to "id" marker when present
    col = re.sub(r"(code|identifier|identifierid|val|value)$","", col)
    # remove trailing numeric noise
    col = re.sub(r"\d+$", "", col)
    return col

def fuzzy_score(a: str, b: str) -> float:
    """
    Simple heuristic similarity score between canon keys a and b.
    Combines longest common prefix, token overlap and common substring length.
    Returns 0..1 float.
    """
    if not a or not b:
        return 0.0
    # longest common prefix bonus
    prefix_len = len(os.path.commonprefix([a, b]))
    lp = prefix_len / max(len(a), len(b))
    # character overlap ratio
    set_a = set(a)
    set_b = set(b)
    overlap = len(set_a & set_b) / max(1, len(set_a | set_b))
    # longest common substring (approx)
    def longest_common_substr(s1, s2):
        m = [[0]*(1+len(s2)) for _ in range(1+len(s1))]
        longest = 0
        for i in range(1, 1+len(s1)):
            for j in range(1, 1+len(s2)):
                if s1[i-1] == s2[j-1]:
                    m[i][j] = m[i-1][j-1] + 1
                    if m[i][j] > longest:
                        longest = m[i][j]
                else:
                    m[i][j] = 0
        return longest
    lcs = longest_common_substr(a, b) / max(1, min(len(a), len(b)))
    # combine with weights
    score = (0.45 * lp) + (0.35 * overlap) + (0.20 * lcs)
    return score

def fuzzy_match_column(col: str, target_list: list) -> str or None:
    """
    Fuzzy-match a column name to one of the official spec fields.
    Returns the spec field (exact spelling from spec_df) if match found, else None.
    """
    canon = canonicalize(col)
    # Tier 1: exact canonical match
    for t in target_list:
        if canonicalize(t) == canon:
            return t
    # Tier 2: quick substring / token exact matches
    tokens = re.split(r"[_\-\s\.]+", str(col).lower())
    for t in target_list:
        tkns = re.split(r"[_\-\s\.]+", str(t).lower())
        if any(tok in t.lower() for tok in tokens if len(tok) > 2):
            return t
    # Tier 3: fuzzy scoring
    best = None
    best_score = 0.0
    for t in target_list:
        tc = canonicalize(t)
        score = fuzzy_score(canon, tc)
        if score > best_score:
            best_score = score
            best = t
    # threshold: require a reasonably high score
    if best_score >= 0.45:
        return best
    return None

def apply_fuzzy_mapping(feed_df: pd.DataFrame, spec_df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames feed_df columns to official spec field names using fuzzy matching.
    If multiple feed columns map to the same spec field, it keeps the first and appends suffix to others.
    """
    spec_cols = spec_df['Attribute'].tolist()
    mapping = {}
    used_targets = set()

    for c in list(feed_df.columns):
        match = fuzzy_match_column(c, spec_cols)
        if match and match not in used_targets:
            mapping[c] = match
            used_targets.add(match)
        elif match and match in used_targets:
            # collision: pick unique name (keep original but also prefix with match)
            mapping[c] = f"{match}__alt"
        else:
            mapping[c] = c  # no match found, keep original

    # apply mapping
    new_df = feed_df.rename(columns=mapping)
    return new_df
# PART 2 of 4: Validation engine, per-product checks, coverage and reporting helpers

# -------------------------
# Requirement parsing util
# -------------------------
def parse_requirement(req):
    """Normalize Requirement column into Required / Recommended / Optional"""
    if pd.isna(req):
        return "Optional"
    s = str(req).strip().lower()
    if "required" in s:
        return "Required"
    if "recommend" in s:
        return "Recommended"
    return "Optional"


# -------------------------
# Type checking helper
# -------------------------
def check_type(series: pd.Series, dtype_str: str):
    """
    Validate a pandas Series against a dtype hint from the spec.
    Returns (ok: bool, detail_message: str)
    """
    if dtype_str is None or pd.isna(dtype_str) or str(dtype_str).strip() == "":
        return True, ""
    s = str(dtype_str).lower()
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return True, ""
    # Integer
    if ("int" in s and "float" not in s) or ("integer" in s):
        coerced = pd.to_numeric(non_null, errors="coerce")
        if coerced.isna().any():
            return False, "Contains non-integer values"
        # check non-negative if spec mentions days/quantity? left generic
        return True, ""
    # Numeric / float
    if ("number" in s) or ("float" in s) or ("price" in s) or ("percentage" in s):
        # strip non-numeric symbols (like % or currency) before coercion
        cleaned = non_null.str.replace(r"[^\d\.\-]", "", regex=True)
        coerced = pd.to_numeric(cleaned, errors="coerce")
        if coerced.isna().any():
            return False, "Contains non-numeric values"
        return True, ""
    # Date-like
    if "date" in s or "iso" in s or "datetime" in s:
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
            return False, "Contains values not parseable as dates"
        return True, ""
    # URL-like
    if "url" in s:
        pattern = re.compile(r"^https?://", re.I)
        ok = non_null.apply(lambda x: bool(pattern.search(x)))
        if not ok.all():
            return False, "Some values do not start with http(s)://"
        return True, ""
    # Enum or string default: no strict check here
    return True, ""


# -------------------------
# Supported-values checker
# -------------------------
def check_supported_values(series: pd.Series, supported_str: str):
    """
    Check that non-null values are within supported values list (if provided).
    supported_str can be comma|pipe|semicolon separated.
    """
    if pd.isna(supported_str) or str(supported_str).strip() == "":
        return True, ""
    allowed = [a.strip().lower() for a in re.split(r"[,|;]", str(supported_str)) if a.strip() != ""]
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


# -------------------------
# Validation rules parser
# -------------------------
def apply_validation_rules(series: pd.Series, rules_text: str):
    """
    Parse and apply simple validation rules described in the spec Validation Rules column.
    Supports: max N chars, min N chars, regex:..., unique:yes, must be <= another field (not implemented), etc.
    Returns (ok:bool, details:str)
    """
    if pd.isna(rules_text) or str(rules_text).strip() == "":
        return True, ""
    parts = [p.strip() for p in str(rules_text).split(";") if p.strip() != ""]
    details = []
    s = series.dropna().astype(str)
    for p in parts:
        low = p.lower()
        # max N
        if "max" in low:
            m = re.search(r"max\s*\D*(\d+)", p)
            if m:
                mx = int(m.group(1))
                too_long = s.apply(len) > mx
                if too_long.any():
                    details.append(f"{too_long.sum()} values exceed max length {mx}")
        # min N
        if "min" in low:
            m = re.search(r"min\s*\D*(\d+)", p)
            if m:
                mn = int(m.group(1))
                too_short = s.apply(len) < mn
                if too_short.any():
                    details.append(f"{too_short.sum()} values shorter than min length {mn}")
        # regex
        if low.startswith("regex:") or "regex:" in low:
            # extract pattern after regex:
            pattern = p.split("regex:", 1)[1]
            try:
                pat = re.compile(pattern)
                bad = ~s.apply(lambda x: bool(pat.search(x)))
                if bad.any():
                    details.append(f"{bad.sum()} values do not match regex")
            except Exception as e:
                details.append(f"Invalid regex: {e}")
        # unique
        if "unique" in low:
            if "yes" in low or "true" in low or ":yes" in low:
                dup = s.duplicated().sum()
                if dup > 0:
                    details.append(f"{dup} duplicate values (should be unique)")
    ok = len(details) == 0
    return ok, "; ".join(details)


# -------------------------
# Dependency evaluator (best-effort)
# -------------------------
def evaluate_dependencies(record_flat: dict, dependencies_text: str):
    """
    Best-effort dependency evaluator for per-product checks.
    Looks for simple patterns like 'Required if gtin missing' or 'Required if availability=preorder'
    Returns (triggered: bool, note: str)
    """
    if pd.isna(dependencies_text) or str(dependencies_text).strip() == "":
        return False, ""
    txt = str(dependencies_text).lower()
    notes = []
    # simple patterns
    m = re.search(r"required if (\w+)\s*(=|is|missing|absent)\s*([^\s,]+)?", txt)
    if m:
        field = m.group(1)
        cond = m.group(2)
        val = m.group(3) if len(m.groups()) >= 3 else None
        # check presence or value in record_flat (case-insensitive keys)
        keys = {k.lower(): k for k in record_flat.keys()}
        if field.lower() in keys:
            actual = record_flat.get(keys[field.lower()])
            if cond in ("missing", "absent") and (actual is None or str(actual).strip() == ""):
                notes.append(f"Dependency triggered: {field} is missing/absent")
            elif val and str(actual).lower() == val.lower():
                notes.append(f"Dependency triggered: {field} == {val}")
        else:
            # if field missing in product, dependency might be triggered (e.g., mpn required if gtin missing)
            if cond in ("missing", "absent"):
                notes.append(f"Dependency note: {field} not present; dependency may apply")
    return (len(notes) > 0), "; ".join(notes)


# -------------------------
# Per-product validator (row-level)
# -------------------------
def validate_product_record(record: dict, spec_df: pd.DataFrame):
    """
    record: flattened dict for one product (keys as columns in feed_df)
    returns: dict with lists of issues
    """
    issues = {
        "missing": [],
        "empty": [],
        "type_issues": [],
        "value_issues": [],
        "validation_issues": [],
        "dependency_notes": [],
        "extras": []
    }

    # make lookup lowercase->original_key for the record
    record_keys = {k.lower(): k for k in record.keys()}

    spec_attrs = spec_df['Attribute'].tolist()
    spec_norm = [a.lower() for a in spec_attrs]

    # extras: keys present in record but not in spec
    for k in record.keys():
        if k.lower() not in spec_norm:
            issues["extras"].append(k)

    # validate each spec attribute
    for _, srow in spec_df.iterrows():
        attr = srow['Attribute']
        attr_norm = attr.lower()
        requirement = parse_requirement(srow.get('Requirement', ""))
        dtype = srow.get('Data Type', "")
        supported = srow.get('Supported Values', "")
        vrules = srow.get('Validation Rules', "")
        dependencies = srow.get('Dependencies', "")

        # find matching key in record (after fuzzy mapping feed columns should already be renamed to spec attr if matched)
        if attr_norm not in record_keys:
            # missing
            if requirement == "Required":
                issues["missing"].append(attr)
            elif requirement == "Recommended":
                issues["missing"].append(attr + " (recommended)")
            continue

        orig_key = record_keys[attr_norm]
        val = record.get(orig_key)

        # empty
        if val is None or (isinstance(val, str) and str(val).strip() == ""):
            if requirement == "Required":
                issues["empty"].append(attr)
            continue

        # type check on single value
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

        # dependencies per product
        triggered, note = evaluate_dependencies(record, dependencies)
        if triggered and note:
            issues["dependency_notes"].append(note)

    return issues


# -------------------------
# Field coverage summary
# -------------------------
def coverage_summary(feed_df: pd.DataFrame, spec_df: pd.DataFrame):
    """
    Returns a DataFrame listing each spec attribute and whether it's present in the feed,
    % filled, and an example value.
    """
    rows = []
    spec_list = spec_df['Attribute'].tolist()
    feed_cols = list(feed_df.columns)
    feed_cols_norm = [c.lower() for c in feed_cols]

    for attr in spec_list:
        a_norm = attr.lower()
        present = a_norm in feed_cols_norm
        filled_pct = 0.0
        example = ""
        if present:
            orig = feed_cols[feed_cols_norm.index(a_norm)]
            filled_pct = 100.0 - feed_df[orig].isna().mean() * 100.0
            non_nulls = feed_df[orig].dropna().astype(str)
            example = non_nulls.iloc[0] if not non_nulls.empty else ""
        rows.append({
            "Attribute": attr,
            "Present": present,
            "% Filled": f"{filled_pct:.1f}%" if present else "0.0%",
            "Example Value": example
        })

    # extras
    spec_set = set([s.lower() for s in spec_list])
    extras = [c for c in feed_df.columns if c.lower() not in spec_set]
    if extras:
        rows.append({
            "Attribute": "(Extra fields)",
            "Present": True,
            "% Filled": "",
            "Example Value": ", ".join(extras[:20])
        })

    return pd.DataFrame(rows)


# -------------------------
# Attribute-level validation (full feed)
# -------------------------
def validate_feed_attribute_level(spec_df: pd.DataFrame, feed_df: pd.DataFrame):
    """
    Validate each attribute across the entire feed and return a report DataFrame.
    """
    spec_df = spec_df.copy()
    spec_df['Attribute_norm'] = spec_df['Attribute'].astype(str).str.strip().str.lower()
    feed_cols = list(feed_df.columns)
    feed_cols_norm = [c.lower() for c in feed_cols]

    report = []
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
            report.append({
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
        report.append({
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
        report.append({
            "Attribute": "(Extra fields)",
            "Requirement": "",
            "Exists in Feed": "Yes",
            "Status": f"⚠️ Extra fields ({len(extras)})",
            "Details": f"Extra / unrecognized fields: {extras[:20]}",
            "Description": "Fields present in feed but not specified",
            "Example": ""
        })

    return pd.DataFrame(report)


# -------------------------
# Batch product-level validation across feed
# -------------------------
def validate_all_products(feed_df: pd.DataFrame, spec_df: pd.DataFrame):
    """
    Iterate through all records and run validate_product_record.
    Returns a DataFrame summarizing issues per record (suitable for CSV export).
    """
    rows = []
    for idx, item in feed_df.iterrows():
        # flattened_record is the dict of this row's columns
        flattened_record = {k: item[k] for k in feed_df.columns}
        issues = validate_product_record(flattened_record, spec_df)

        prod_id = flattened_record.get('id') or flattened_record.get('ID') or flattened_record.get('Id') or ""
        rows.append({
            "record_index": int(idx),
            "id": prod_id,
            "missing": "|".join(issues["missing"]),
            "empty": "|".join(issues["empty"]),
            "type_issues": "|".join(issues["type_issues"]),
            "value_issues": "|".join(issues["value_issues"]),
            "validation_issues": "|".join(issues["validation_issues"]),
            "dependency_notes": "|".join(issues["dependency_notes"]),
            "extras": "|".join(issues["extras"]),
            "has_issues": any(len(issues[k]) > 0 for k in issues)
        })
    return pd.DataFrame(rows)
# PART 3 of 4: Streamlit UI + file upload handling + fuzzy mapping + expanders

# -------------------------
# STREAMLIT UI
# -------------------------
st.title("🔎 ChatGPT Product Feed Validator — Full Schema + Fuzzy Mapping")
st.write("""
This tool validates JSON/XML product feeds against the embedded ChatGPT Product Feed Spec.

### ✅ Features
- Auto-flatten JSON & XML
- Auto-schema detection
- **Fuzzy header mapping** (e.g., SKU → id, brandName → brand, main_image → image_link)
- Attribute-level validation
- Full product-level validation (expandable UI)
- Field coverage matrix
- Downloadable CSV reports
- Handles large feeds safely
""")

st.sidebar.header("⚙️ Settings")
max_expand = st.sidebar.number_input(
    "Max Products to Expand in UI", min_value=10, max_value=5000, value=250, step=10
)
allow_all = st.sidebar.checkbox("Allow unlimited expanders (slow for large feeds)", value=False)

st.sidebar.markdown("---")
st.sidebar.write("Upload a JSON or XML feed to begin.")

uploaded = st.file_uploader("📤 Upload Product Feed (JSON or XML)", type=["json", "xml"])

if uploaded is None:
    st.info("Upload a feed file to start validation.")
else:
    try:
        uploaded.seek(0)
        ext = uploaded.name.lower().split(".")[-1]

        # -------------------------
        # 1. Parse & Flatten Feed
        # -------------------------
        with st.spinner("Parsing feed..."):
            if ext == "json":
                feed_df = json_to_records(uploaded)
            else:
                feed_df = xml_to_records(uploaded)

        if feed_df.empty:
            st.error("❌ Parsing failed — no records found. Check feed structure.")
            st.stop()

        st.success(f"Successfully parsed {len(feed_df)} products.")

        # -------------------------------
        # 2. Apply Fuzzy Header Mapping
        # -------------------------------
        with st.spinner("Applying fuzzy header mapping…"):
            feed_df = apply_fuzzy_mapping(feed_df, spec_df)

        st.info("🔁 Column names mapped to closest spec fields using fuzzy logic.")

        # Normalize ALL column names to lowercase (spec is lowercase)
        feed_df.columns = [str(c).strip() for c in feed_df.columns]

        # -------------------------
        # 3. Attribute-Level Validation
        # -------------------------
        st.markdown("## 1️⃣ Attribute-Level Validation Report")
        with st.spinner("Validating attributes across the full feed…"):
            attr_report = validate_feed_attribute_level(spec_df, feed_df)

        st.dataframe(attr_report, use_container_width=True)
        st.download_button(
            "⬇️ Download Attribute-Level Report (CSV)",
            attr_report.to_csv(index=False).encode("utf-8"),
            "attribute_level_report.csv",
            "text/csv",
        )

        # -------------------------
        # 4. Field Coverage Matrix
        # -------------------------
        st.markdown("## 2️⃣ Field Coverage Summary")
        with st.spinner("Computing field coverage…"):
            coverage = coverage_summary(feed_df, spec_df)

        st.dataframe(coverage, use_container_width=True)
        st.download_button(
            "⬇️ Download Field Coverage CSV",
            coverage.to_csv(index=False).encode("utf-8"),
            "field_coverage.csv",
            "text/csv",
        )

        # -------------------------
        # 5. Product-Level Validation
        # -------------------------
        st.markdown("## 3️⃣ Product-Level Validation (Expandable)")
        with st.spinner("Validating each individual product…"):
            product_report = validate_all_products(feed_df, spec_df)

        st.success("Product-level validation completed!")

        # Show summary table
        st.markdown("### Product-Level Summary Table")
        st.dataframe(product_report.head(200), use_container_width=True)
        st.download_button(
            "⬇️ Download Full Product-Level Issues CSV",
            product_report.to_csv(index=False).encode("utf-8"),
            "product_level_issues.csv",
            "text/csv",
        )

        # --------------------------------
        # 6. Expandable Per-Product Panels
        # --------------------------------
        st.markdown("## 4️⃣ Inspect Products (Expandable Panels)")

        if allow_all:
            expand_limit = len(feed_df)
        else:
            expand_limit = min(len(feed_df), max_expand)

        st.info(f"Showing first **{expand_limit} products** (use sidebar to increase).")

        for idx in range(expand_limit):
            row = feed_df.iloc[idx]
            flattened_record = {k: row[k] for k in feed_df.columns}

            # Get issues for this product
            issues = validate_product_record(flattened_record, spec_df)

            # Create header text
            product_id = flattened_record.get("id") or ""
            issue_summary = []

            if issues["missing"]:
                issue_summary.append(f"Missing: {', '.join(issues['missing'][:5])}")
            if issues["empty"]:
                issue_summary.append(f"Empty: {', '.join(issues['empty'][:5])}")
            if issues["type_issues"]:
                issue_summary.append(f"Type: {len(issues['type_issues'])} issues")
            if issues["value_issues"]:
                issue_summary.append(f"Value: {len(issues['value_issues'])} issues")
            if issues["validation_issues"]:
                issue_summary.append(f"Rules: {len(issues['validation_issues'])} issues")
            if issues["extras"]:
                issue_summary.append(f"Extras: {len(issues['extras'])}")

            summary_text = " | ".join(issue_summary) if issue_summary else "No issues"

            header = f"Product #{idx}"
            if product_id:
                header += f" — id: {product_id}"
            header += f" — {summary_text}"

            with st.expander(header, expanded=False):
                st.markdown("### Issues Breakdown")
                st.json(issues)

                st.markdown("### Raw Flattened Fields")
                st.dataframe(
                    pd.DataFrame([flattened_record]).T.rename(columns={0: "value"}),
                    use_container_width=True,
                )

        st.success("Validation completed successfully!")

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.exception(e)
# PART 4 of 4: Helper utilities, row-level failure extractor, final UI bits, and closing notes

# -------------------------
# Row-level failure extraction (detailed per-attribute per-row)
# -------------------------
def extract_row_level_failures(feed_df: pd.DataFrame, spec_df: pd.DataFrame):
    """
    Returns a DataFrame listing per-row failures:
    columns: record_index, id, attribute, issue_type, details, value
    issue_type in [missing, empty, type, value, validation]
    """
    rows = []
    spec_list = spec_df['Attribute'].tolist()
    feed_cols = list(feed_df.columns)
    feed_cols_norm = [c.lower() for c in feed_cols]

    for idx, item in feed_df.iterrows():
        flattened_record = {k: item[k] for k in feed_df.columns}
        rec_keys_lower = {k.lower(): k for k in flattened_record.keys()}
        prod_id = flattened_record.get('id') or flattened_record.get('ID') or ""

        for _, srow in spec_df.iterrows():
            attr = srow['Attribute']
            a_norm = attr.lower()
            requirement = parse_requirement(srow.get('Requirement', ""))
            dtype = srow.get('Data Type', "")
            supported = srow.get('Supported Values', "")
            vrules = srow.get('Validation Rules', "")

            # If column missing from feed entirely
            if a_norm not in feed_cols_norm:
                if requirement == "Required":
                    rows.append({
                        "record_index": int(idx),
                        "id": prod_id,
                        "attribute": attr,
                        "issue_type": "missing",
                        "details": "Attribute not present in feed",
                        "value": ""
                    })
                # recommended/optional absences are not per-row issues (they're structural)
                continue

            orig_col = feed_cols[feed_cols_norm.index(a_norm)]
            val = flattened_record.get(orig_col)

            # missing/empty
            if val is None or (isinstance(val, str) and str(val).strip() == ""):
                if requirement == "Required":
                    rows.append({
                        "record_index": int(idx),
                        "id": prod_id,
                        "attribute": attr,
                        "issue_type": "empty",
                        "details": "Required value empty or null",
                        "value": "" if val is None else str(val)
                    })
                continue

            # type issues
            ok_type, tdet = check_type(pd.Series([val]), dtype)
            if not ok_type:
                rows.append({
                    "record_index": int(idx),
                    "id": prod_id,
                    "attribute": attr,
                    "issue_type": "type",
                    "details": tdet,
                    "value": str(val)
                })

            # supported values
            ok_vals, vdet = check_supported_values(pd.Series([val]), supported)
            if not ok_vals:
                rows.append({
                    "record_index": int(idx),
                    "id": prod_id,
                    "attribute": attr,
                    "issue_type": "value",
                    "details": vdet,
                    "value": str(val)
                })

            # validation rule failures
            ok_rules, rdet = apply_validation_rules(pd.Series([val]), vrules)
            if not ok_rules:
                rows.append({
                    "record_index": int(idx),
                    "id": prod_id,
                    "attribute": attr,
                    "issue_type": "validation",
                    "details": rdet,
                    "value": str(val)
                })
    if not rows:
        return pd.DataFrame(columns=["record_index","id","attribute","issue_type","details","value"])
    return pd.DataFrame(rows)


# -------------------------
# Optional: fuzzy-mapper tuning endpoint (expose mapping preview)
# -------------------------
def preview_fuzzy_mapping(feed_df: pd.DataFrame, spec_df: pd.DataFrame):
    """
    Returns a small DataFrame mapping original feed columns -> mapped spec columns
    """
    spec_cols = spec_df['Attribute'].tolist()
    mapping = []
    for c in list(feed_df.columns):
        match = fuzzy_match_column(c, spec_cols)
        mapped = match if match else "(no match)"
        mapping.append({"original": c, "mapped_to": mapped})
    return pd.DataFrame(mapping)


# -------------------------
# FINAL UI: Row-level failures + Mapping preview
# -------------------------
st.markdown("---")
st.markdown("## 5️⃣ Advanced: Row-level failures & mapping preview")

col1, col2 = st.columns([1,1])
with col1:
    if st.button("🔎 Generate Row-level Failure CSV (detailed)"):
        with st.spinner("Extracting row-level failures (this scans every cell)..."):
            failures_df = extract_row_level_failures(feed_df, spec_df)
        if failures_df.empty:
            st.success("No row-level failures found across scanned attributes.")
        else:
            st.write(f"Found {len(failures_df)} row-level issues — showing first 200 rows")
            st.dataframe(failures_df.head(200), use_container_width=True)
            st.download_button(
                "⬇️ Download row-level failures (CSV)",
                failures_df.to_csv(index=False).encode("utf-8"),
                "row_level_failures.csv",
                "text/csv",
            )

with col2:
    if st.button("🔁 Preview Fuzzy Column Mapping"):
        with st.spinner("Computing column mapping preview..."):
            mapping_preview = preview_fuzzy_mapping(feed_df, spec_df)
        st.dataframe(mapping_preview, use_container_width=True)
        st.download_button(
            "⬇️ Download mapping preview (CSV)",
            mapping_preview.to_csv(index=False).encode("utf-8"),
            "mapping_preview.csv",
            "text/csv",
        )

st.markdown("---")
st.info("Tips: use the mapping preview to confirm fuzzy mappings. If a mapping is incorrect, rename columns in your source feed or update field names prior to upload. You can also inspect the row-level failures CSV in a spreadsheet to see exact failing values.")

# -------------------------
# SMALL UTILITY: Allow the user to re-run with adjusted mapping
# -------------------------
st.markdown("## 6️⃣ Advanced: Remap columns manually (optional)")

if st.checkbox("Manually remap columns before validation", value=False):
    st.write("Map feed columns to spec fields. Leave blank to keep original mapping.")
    # Show current columns and allow user input boxes for new mapping
    current_cols = list(feed_df.columns)
    remap = {}
    for c in current_cols:
        new_name = st.text_input(f"Map `{c}` to spec field (leave blank to keep):", value="")
        if new_name and new_name.strip() != "":
            remap[c] = new_name.strip()
    if remap:
        if st.button("Apply manual remapping"):
            feed_df.rename(columns=remap, inplace=True)
            st.success("Manual remapping applied. Re-run validation from top if needed.")
            st.experimental_rerun()

st.markdown("### Done — validator ready")
st.write("If you want further enhancements I can add:")
st.write("- HTTP HEAD checks for `link`, `image_link`, `seller_url` (will make network calls).")
st.write("- ISO currency/region strict validation.")
st.write("- Auto-corrections (split price into amount + currency, normalize booleans).")
st.write("- Integrate with OpenAI to provide automatic textual recommendations per product.")

# -------------------------
# requirements.txt (for convenience)
# -------------------------
REQ_TXT = """streamlit
pandas
lxml
"""
st.sidebar.markdown("---")
st.sidebar.subheader("Requirements")
st.sidebar.code(REQ_TXT)

# End of PART 4 (Final)
