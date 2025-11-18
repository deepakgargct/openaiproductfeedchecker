# ============================================================
# PART 1 — IMPORTS, SPEC, PARSING, SMART FUZZY MAPPER
# ============================================================

import streamlit as st
import pandas as pd
import json
import xml.etree.ElementTree as ET
from io import StringIO, BytesIO
import re
from datetime import datetime
import os

st.set_page_config(page_title="Auto-Enriched ChatGPT Feed Validator", layout="wide")

# ============================================================
# 1. EMBEDDED CHATGPT PRODUCT FEED SPEC
# ============================================================
SPEC_CSV = """Attribute\tData Type\tSupported Values\tDescription\tExample\tRequirement\tDependencies\tValidation Rules
enable_search\tEnum\ttrue, false\tControls whether the product can be surfaced in ChatGPT search results.\tTRUE\tRequired\t—\tLower-case string
enable_checkout\tEnum\ttrue, false\tAllows direct purchase...\tTRUE\tRequired\t—\tLower-case string
id\tString (alphanumeric)\t—\tMerchant product ID\tSKU12345\tRequired\t—\tMax 100 chars
gtin\tString (numeric)\tGTIN, UPC, ISBN\tUniversal product identifier\t1.23457E+11\tRecommended\t—\t8–14 digits
mpn\tString (alphanumeric)\t—\tManufacturer part number\tGPT5\tRequired if gtin missing\t—\tMax 70 chars
title\tString\t—\tProduct title\tRunning Shoes\tRequired\t—\tMax 150 chars
description\tString\t—\tFull description\tWaterproof trail shoe\tRequired\t—\tMax 5000 chars
link\tURL\tRFC 1738\tProduct detail page\thttps://example.com\tRequired\t—\tMust be HTTP 200
condition\tEnum\tnew, refurbished, used\tCondition\tnew\tRequired if differs\t—\tLower-case string
product_category\tString\t—\tCategory taxonomy\tShoes\tRequired\t—\tUse “>”
brand\tString\t—\tBrand\tOpenAI\tRequired\t—\tMax 70 chars
material\tString\t—\tMaterial\tLeather\tRequired\t—\tMax 100 chars
dimensions\tString\t—\tDimensions\t10x5x3 in\tOptional\t—\tUnits required
length\tNumber + unit\t—\tLength\t10 mm\tOptional\tProvide 3 dims\tUnits required
width\tNumber + unit\t—\tWidth\t10 mm\tOptional\tProvide 3 dims\tUnits required
height\tNumber + unit\t—\tHeight\t10 mm\tOptional\tProvide 3 dims\tUnits required
weight\tNumber + unit\t—\tWeight\t1.5 lb\tRequired\t—\tPositive number
age_group\tEnum\tnewborn, infant, toddler, kids, adult\tTarget demographic\tadult\tOptional\t—\tLower-case string
image_link\tURL\tRFC 1738\tMain image\thttps://example.com/img.jpg\tRequired\t—\tJPEG/PNG
additional_image_link\tURL array\tRFC 1738\tExtra images\t...\tOptional\t—\tComma separated
video_link\tURL\tRFC 1738\tProduct video\thttps://... \tOptional\t—\tMust be public
model_3d_link\tURL\tRFC 1738\t3D model\t...\tOptional\t—\tGLB preferred
price\tNumber + currency\tISO 4217\tRegular price\t79.99 USD\tRequired\t—\tMust include currency
sale_price\tNumber + currency\tISO 4217\tSale price\t59.99 USD\tOptional\t—\tMust be <= price
availability\tEnum\tin_stock, out_of_stock, preorder\tAvailability\tin_stock\tRequired\t—\tLower-case
inventory_quantity\tInteger\t—\tStock count\t25\tRequired\t—\t>=0
seller_name\tString\t—\tSeller name\tExample Store\tRequired\t—\tMax 70 chars
seller_url\tURL\tRFC 1738\tSeller page\thttps://example.com\tRequired\t—\tHTTPS preferred
seller_privacy_policy\tURL\tRFC 1738\tPrivacy policy\thttps://...\tRequired if checkout\t—\tHTTPS
seller_tos\tURL\tRFC 1738\tTerms of service\thttps://...\tRequired if checkout\t—\tHTTPS
return_policy\tURL\tRFC 1738\tReturn policy\thttps://...\tRequired\t—\tHTTPS
return_window\tInteger\tDays\tNumber of days\t30\tRequired\t—\tPositive
"""

@st.cache_data
def load_spec_df():
    return pd.read_csv(StringIO(SPEC_CSV), sep="\t")

spec_df = load_spec_df()

# ============================================================
# 2. JSON & XML PARSING HELPERS
# ============================================================
def _flatten_json(obj, prefix="", out=None):
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{prefix}.{k}" if prefix else k
            _flatten_json(v, new_key, out)
    elif isinstance(obj, list):
        if all(not isinstance(i, (dict, list)) for i in obj):
            out[prefix] = "|".join([str(i) for i in obj])
        else:
            for idx, item in enumerate(obj):
                new_key = f"{prefix}.{idx}"
                _flatten_json(item, new_key, out)
    else:
        out[prefix] = obj
    return out

def json_to_records_with_raw(file_bytes):
    file_bytes.seek(0)
    raw_text = file_bytes.read().decode("utf-8", errors="ignore")
    data = json.loads(raw_text)

    rows = []
    raw_items = []

    if isinstance(data, list):
        for item in data:
            raw_items.append(item)
            rows.append(_flatten_json(item))
        return pd.DataFrame(rows), raw_items

    if isinstance(data, dict):
        if any(isinstance(v, list) for v in data.values()):
            for k, v in data.items():
                if isinstance(v, list) and v:
                    return pd.DataFrame([_flatten_json(i) for i in v]), v
        raw_items = [data]
        return pd.DataFrame([_flatten_json(data)]), raw_items

    return pd.DataFrame(), []

def xml_to_dict(elem):
    d = {}
    d.update(elem.attrib)
    children = list(elem)
    if children:
        grouped = {}
        for ch in children:
            grouped.setdefault(ch.tag, []).append(xml_to_dict(ch))
        for k, v in grouped.items():
            d[k] = v if len(v) > 1 else v[0]
    text = (elem.text or "").strip()
    if text and not d:
        return text
    if text:
        d["_text"] = text
    return d

def xml_to_records(file_bytes):
    file_bytes.seek(0)
    text = file_bytes.read().decode("utf-8", errors="ignore")
    root = ET.fromstring(text)
    rows = []
    for child in root:
        rows.append(_flatten_json(xml_to_dict(child)))
    if not rows:
        rows.append(_flatten_json(xml_to_dict(root)))
    return pd.DataFrame(rows)

# ============================================================
# 3. SMART URL EXTRACTION (for images/media arrays)
# ============================================================
url_re = re.compile(r"https?://[^\s\"']+", re.IGNORECASE)

def extract_urls_from_any(x):
    urls = []
    if x is None:
        return urls
    if isinstance(x, str):
        urls.extend(url_re.findall(x))
        return urls
    if isinstance(x, list):
        for v in x:
            urls.extend(extract_urls_from_any(v))
        return list(dict.fromkeys(urls))  # unique
    if isinstance(x, dict):
        for key in ["src", "url", "href", "image", "link"]:
            if key in x and isinstance(x[key], str):
                if url_re.search(x[key]):
                    urls.append(x[key])
        for v in x.values():
            urls.extend(extract_urls_from_any(v))
        return list(dict.fromkeys(urls))
    return urls

# ============================================================
# 4. SMART FUZZY MAPPER (TITLE vs DESCRIPTION LOGIC INCLUDED)
# ============================================================

def canonicalize(col):
    c = str(col).lower().strip()
    c = re.sub(r"[^a-z0-9]+", "", c)
    return c

def fuzzy_score(a, b):
    if not a or not b:
        return 0.0
    prefix_len = len(os.path.commonprefix([a, b]))
    lp = prefix_len / max(len(a), len(b))
    set_a, set_b = set(a), set(b)
    overlap = len(set_a & set_b) / max(1, len(set_a | set_b))

    def lcs(s1, s2):
        m = [[0]*(1+len(s2)) for _ in range(1+len(s1))]
        longest = 0
        for i in range(1, len(s1)+1):
            for j in range(1, len(s2)+1):
                if s1[i-1] == s2[j-1]:
                    m[i][j] = m[i-1][j-1] + 1
                    longest = max(longest, m[i][j])
        return longest

    lcs_val = lcs(a, b) / max(1, min(len(a), len(b)))
    return (0.45 * lp) + (0.35 * overlap) + (0.20 * lcs_val)

def detect_free_text_column(series: pd.Series):
    """
    Detect if column contains long, diverse text → description-like.
    """
    vals = series.dropna().astype(str).head(50).tolist()
    if len(vals) < 5:
        return False, False  # not enough data

    avg_len = sum(len(v) for v in vals) / len(vals)
    distinct_pct = len(set(vals)) / len(vals)

    is_description_like = avg_len > 40 or distinct_pct > 0.8
    is_title_like = (10 < avg_len < 40) and (distinct_pct > 0.5)

    return is_title_like, is_description_like

def fuzzy_match_column(col, target_fields, feed_df=None):
    """
    Smart fuzzy mapper with protection:
    - Detect free-text cols and assign to title/description appropriately
    - Avoid mapping these to enum fields
    - Higher threshold to reduce errors
    """
    col_canon = canonicalize(col)

    # Check free-text profile
    is_title_like = False
    is_description_like = False
    if feed_df is not None and col in feed_df.columns:
        series = feed_df[col]
        is_title_like, is_description_like = detect_free_text_column(series)

    # Forced mapping if column name explicitly contains these
    if "title" in col.lower() and not is_description_like:
        return "title"
    if "description" in col.lower() or "body_html" in col.lower():
        return "description"

    # If description-like, map to description only
    if is_description_like:
        return "description"

    # If title-like, map to title
    if is_title_like:
        return "title"

    # Avoid matching free-text to enum fields
    enum_fields = {"condition","gender","age_group","availability","pickup_method",
                   "relationship_type"}

    # Exact canonical match
    for t in target_fields:
        if canonicalize(t) == col_canon:
            return t

    # Token similarity
    tokens = re.split(r"[_\-\s\.]+", col.lower())
    for t in target_fields:
        t_tokens = re.split(r"[_\-\s\.]+", t.lower())
        if any(tok in t.lower() for tok in tokens if len(tok) > 2):
            return t

    # Fuzzy scoring
    best = None
    best_score = 0.0
    for t in target_fields:
        if t.lower() in enum_fields and (is_title_like or is_description_like):
            continue  # avoid enum mismatch
        score = fuzzy_score(col_canon, canonicalize(t))
        if score > best_score:
            best = t
            best_score = score

    if best_score >= 0.55:
        return best
    return None

def apply_fuzzy_mapping(df, spec_df):
    targets = spec_df["Attribute"].tolist()
    mapping = {}
    for c in df.columns:
        match = fuzzy_match_column(c, targets, df)
        mapping[c] = match if match else c
    return df.rename(columns=mapping)
# ============================================================
# PART 2 — Validation helpers, enrichment, and per-product checks
# ============================================================

# -------------------------
# Requirement parser
# -------------------------
def parse_requirement(req):
    if pd.isna(req) or req is None:
        return "Optional"
    s = str(req).lower()
    if "required" in s:
        return "Required"
    if "recommend" in s:
        return "Recommended"
    return "Optional"

# -------------------------
# Type checker
# -------------------------
def check_type(series: pd.Series, dtype_hint: str):
    """
    Validate values in a Series against a dtype hint from the spec.
    Returns (ok: bool, message: str)
    """
    if dtype_hint is None or pd.isna(dtype_hint) or str(dtype_hint).strip() == "":
        return True, ""
    s = str(dtype_hint).lower()
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return True, ""
    # Integer
    if ("int" in s and "float" not in s) or ("integer" in s):
        coerced = pd.to_numeric(non_null, errors="coerce")
        if coerced.isna().any():
            return False, "Contains non-integer values"
        return True, ""
    # Numeric
    if ("number" in s) or ("float" in s) or ("price" in s) or ("percentage" in s):
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
        ok = non_null.apply(lambda x: bool(re.search(r"^https?://", x, re.I)))
        if not ok.all():
            return False, "Some values do not start with http(s)://"
        return True, ""
    # Default: assume string OK
    return True, ""

# -------------------------
# Supported values checker
# -------------------------
def check_supported_values(series: pd.Series, supported_str: str):
    if pd.isna(supported_str) or not supported_str:
        return True, ""
    allowed = [a.strip().lower() for a in re.split(r"[,|;]", str(supported_str)) if a.strip()]
    if not allowed:
        return True, ""
    non_null = series.dropna().astype(str).str.lower()
    if non_null.empty:
        return True, ""
    bad = ~non_null.isin(allowed)
    if bad.any():
        examples = list(non_null[bad].unique()[:6])
        return False, f"Values outside allowed set. Examples: {examples}"
    return True, ""

# -------------------------
# Validation rules parser
# -------------------------
def apply_validation_rules(series: pd.Series, rules_text: str):
    if pd.isna(rules_text) or not rules_text:
        return True, ""
    parts = [p.strip() for p in str(rules_text).split(";") if p.strip()]
    details = []
    s = series.dropna().astype(str)
    for p in parts:
        low = p.lower()
        if "max" in low:
            m = re.search(r"max\s*\D*(\d+)", p)
            if m:
                mx = int(m.group(1))
                too_long = s.apply(len) > mx
                if too_long.any():
                    details.append(f"{too_long.sum()} values exceed max length {mx}")
        if "min" in low:
            m = re.search(r"min\s*\D*(\d+)", p)
            if m:
                mn = int(m.group(1))
                too_short = s.apply(len) < mn
                if too_short.any():
                    details.append(f"{too_short.sum()} values shorter than min length {mn}")
        if "regex:" in low:
            pattern = p.split("regex:",1)[1]
            try:
                pat = re.compile(pattern)
                bad = ~s.apply(lambda x: bool(pat.search(x)))
                if bad.any():
                    details.append(f"{bad.sum()} values do not match regex")
            except Exception as e:
                details.append(f"Invalid regex: {e}")
        if "unique" in low:
            if "yes" in low or "true" in low:
                dup = s.duplicated().sum()
                if dup > 0:
                    details.append(f"{dup} duplicate values (should be unique)")
    ok = len(details) == 0
    return ok, "; ".join(details)

# -------------------------
# Enrichment: Shopify & general feed normalization
# -------------------------
def strip_html_tags(text):
    if not isinstance(text, str):
        return text
    return re.sub(r"<[^>]+>", "", text).strip()

def enrich_record_from_shopify(raw_item: dict, flat_record: dict, default_currency="USD"):
    """
    Create enriched record (dict) combining flattened values and best-effort Shopify mappings.
    """
    enriched = dict(flat_record)  # copy existing flattened values

    # title
    if not enriched.get("title"):
        if raw_item.get("title"):
            enriched["title"] = raw_item.get("title")

    # description/body_html -> description (strip tags)
    if not enriched.get("description"):
        bh = raw_item.get("body_html") or raw_item.get("description") or raw_item.get("html_description")
        if bh:
            enriched["description"] = strip_html_tags(bh)

    # brand: vendor
    if not enriched.get("brand"):
        if raw_item.get("vendor"):
            enriched["brand"] = raw_item.get("vendor")

    # product_category: product_type or tags
    if not enriched.get("product_category"):
        if raw_item.get("product_type"):
            enriched["product_category"] = raw_item.get("product_type")
        else:
            tags = raw_item.get("tags")
            if tags:
                if isinstance(tags, list):
                    enriched["product_category"] = ", ".join(tags[:3])
                else:
                    enriched["product_category"] = str(tags)

    # images: collect from images, media, gallery, assets, photos
    images = []
    if isinstance(raw_item.get("images"), list):
        for img in raw_item.get("images"):
            for u in extract_urls_from_any(img):
                if u not in images:
                    images.append(u)
    for alt in ["media","photos","gallery","assets"]:
        if isinstance(raw_item.get(alt), list):
            for val in raw_item.get(alt):
                for u in extract_urls_from_any(val):
                    if u not in images:
                        images.append(u)
    # also scan flattened values that may contain URLs
    for k, v in flat_record.items():
        if isinstance(v, str) and url_re.search(v):
            for u in url_re.findall(v):
                if u not in images:
                    images.append(u)

    if images:
        enriched["image_link"] = images[0]
        if len(images) > 1:
            enriched["additional_image_link"] = ", ".join(images[1:])
        else:
            enriched["additional_image_link"] = ""

    # variants -> price, sale_price, mpn, availability, inventory
    variants = raw_item.get("variants") or []
    if isinstance(variants, list) and variants:
        # choose first variant as canonical for single-value fields
        first = variants[0]
        # price
        if not enriched.get("price"):
            p = first.get("price") or first.get("compare_at_price") or first.get("presentment_price")
            if p is not None:
                pstr = str(p).strip()
                if re.search(r"\b[A-Z]{3}\b", pstr):
                    enriched["price"] = pstr
                else:
                    enriched["price"] = f"{pstr} {default_currency}" if default_currency else pstr
        # sale_price
        if not enriched.get("sale_price"):
            cp = first.get("compare_at_price") or first.get("compare_price")
            if cp:
                cpstr = str(cp).strip()
                if re.search(r"\b[A-Z]{3}\b", cpstr):
                    enriched["sale_price"] = cpstr
                else:
                    enriched["sale_price"] = f"{cpstr} {default_currency}" if default_currency else cpstr
        # mpn from sku
        if not enriched.get("mpn"):
            sku = first.get("sku") or first.get("mpn")
            if sku:
                enriched["mpn"] = sku
        # availability and inventory
        any_avail = False
        total_inv = 0
        inv_found = False
        for v in variants:
            if v.get("available") in (True, "true", "True", 1, "1"):
                any_avail = True
            if v.get("inventory_quantity") is not None:
                try:
                    total_inv += int(v.get("inventory_quantity"))
                    inv_found = True
                except:
                    pass
        if not enriched.get("availability"):
            enriched["availability"] = "in_stock" if any_avail else "out_of_stock"
        if inv_found and not enriched.get("inventory_quantity"):
            enriched["inventory_quantity"] = total_inv

    # link: try to use direct url-like fields if not present
    if not enriched.get("link"):
        for key in ["url","product_url","permalink","handle"]:
            val = raw_item.get(key)
            if val and isinstance(val, str) and url_re.search(val):
                enriched["link"] = val
                break

    # id fallback: prefer existing id, else raw id
    if not enriched.get("id") and raw_item.get("id"):
        enriched["id"] = raw_item.get("id")

    return enriched

# -------------------------
# Per-product validator
# -------------------------
def validate_product_record(record: dict, spec_df: pd.DataFrame):
    issues = {
        "missing": [],
        "empty": [],
        "type_issues": [],
        "value_issues": [],
        "validation_issues": [],
        "dependency_notes": [],
        "extras": []
    }
    record_keys = {k.lower(): k for k in record.keys()}
    spec_attrs = spec_df["Attribute"].tolist()
    spec_norm = [a.lower() for a in spec_attrs]

    # extras
    for k in record.keys():
        if k.lower() not in spec_norm:
            issues["extras"].append(k)

    for _, s in spec_df.iterrows():
        attr = s["Attribute"]
        a_norm = attr.lower()
        requirement = parse_requirement(s.get("Requirement", ""))
        dtype = s.get("Data Type", "")
        supported = s.get("Supported Values", "")
        vrules = s.get("Validation Rules", "")

        if a_norm not in record_keys:
            if requirement == "Required":
                issues["missing"].append(attr)
            elif requirement == "Recommended":
                issues["missing"].append(attr + " (recommended)")
            continue

        orig_key = record_keys[a_norm]
        val = record.get(orig_key)

        if val is None or (isinstance(val, str) and not str(val).strip()):
            if requirement == "Required":
                issues["empty"].append(attr)
            continue

        series = pd.Series([val])
        ok_t, tdet = check_type(series, dtype)
        if not ok_t:
            issues["type_issues"].append(f"{attr}: {tdet}")
        ok_v, vdet = check_supported_values(series, supported)
        if not ok_v:
            issues["value_issues"].append(f"{attr}: {vdet}")
        ok_r, rdet = apply_validation_rules(series, vrules)
        if not ok_r:
            issues["validation_issues"].append(f"{attr}: {rdet}")

    return issues

# -------------------------
# Coverage summary & feed-level validation
# -------------------------
def coverage_summary(feed_df: pd.DataFrame, spec_df: pd.DataFrame):
    rows = []
    spec_list = spec_df["Attribute"].tolist()
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
        rows.append({"Attribute": attr, "Present": present, "% Filled": f"{filled_pct:.1f}%", "Example Value": example})
    spec_set = set([s.lower() for s in spec_list])
    extras = [c for c in feed_df.columns if c.lower() not in spec_set]
    if extras:
        rows.append({"Attribute": "(Extra fields)", "Present": True, "% Filled": "", "Example Value": ", ".join(extras[:20])})
    return pd.DataFrame(rows)

def validate_feed_attribute_level(spec_df: pd.DataFrame, feed_df: pd.DataFrame):
    spec_df = spec_df.copy()
    spec_df["Attribute_norm"] = spec_df["Attribute"].astype(str).str.lower().str.strip()
    feed_cols = list(feed_df.columns)
    feed_cols_norm = [c.lower() for c in feed_cols]
    report = []
    for _, s in spec_df.iterrows():
        attr = s["Attribute"]
        a_norm = s["Attribute_norm"]
        dtype = s.get("Data Type", "")
        supported = s.get("Supported Values", "")
        requirement = parse_requirement(s.get("Requirement", ""))
        vrules = s.get("Validation Rules", "")

        if a_norm not in feed_cols_norm:
            status = "❌ Missing (Required)" if requirement == "Required" else ("⚠️ Missing (Recommended)" if requirement == "Recommended" else "ℹ️ Missing (Optional)")
            report.append({"Attribute": attr, "Requirement": requirement, "Exists in Feed": "No", "Status": status, "Details": "Not present in feed", "Description": s.get("Description",""), "Example": s.get("Example","")})
            continue

        orig_col = feed_cols[feed_cols_norm.index(a_norm)]
        col = feed_df[orig_col]
        details = []
        empty_pct = col.isna().mean() * 100
        if empty_pct > 0:
            details.append(f"{empty_pct:.1f}% empty values")
        ok_t, tdet = check_type(col, dtype)
        if not ok_t:
            details.append(tdet)
        ok_v, vdet = check_supported_values(col, supported)
        if not ok_v:
            details.append(vdet)
        ok_r, rdet = apply_validation_rules(col, vrules)
        if not ok_r:
            details.append(rdet)
        status = "✅ Present & Valid" if not details else "⚠️ Issues"
        report.append({"Attribute": attr, "Requirement": requirement, "Exists in Feed": "Yes", "Status": status, "Details": " | ".join(details) if details else "", "Description": s.get("Description",""), "Example": s.get("Example","")})
    spec_set = set(spec_df["Attribute_norm"].tolist())
    extras = [c for c in feed_cols if c.lower() not in spec_set]
    if extras:
        report.append({"Attribute": "(Extra fields)", "Requirement": "", "Exists in Feed": "Yes", "Status": f"⚠️ Extra fields ({len(extras)})", "Details": f"Extra / unrecognized fields: {extras[:20]}", "Description": "Fields present in feed but not specified", "Example": ""})
    return pd.DataFrame(report)

def validate_all_products(feed_df: pd.DataFrame, spec_df: pd.DataFrame):
    rows = []
    for idx, item in feed_df.iterrows():
        flat = {k: item[k] for k in feed_df.columns}
        issues = validate_product_record(flat, spec_df)
        prod_id = flat.get("id") or flat.get("ID") or flat.get("Id") or ""
        rows.append({"record_index": int(idx), "id": prod_id, "missing": "|".join(issues["missing"]), "empty": "|".join(issues["empty"]), "type_issues": "|".join(issues["type_issues"]), "value_issues": "|".join(issues["value_issues"]), "validation_issues": "|".join(issues["validation_issues"]), "dependency_notes": "|".join(issues["dependency_notes"]), "extras": "|".join(issues["extras"]), "has_issues": any(len(issues[k])>0 for k in issues)})
    return pd.DataFrame(rows)

# -------------------------
# Row-level failures & mapping preview helpers
# -------------------------
def extract_row_level_failures(feed_df: pd.DataFrame, spec_df: pd.DataFrame):
    rows = []
    feed_cols_norm = [c.lower() for c in feed_df.columns]
    for idx, item in feed_df.iterrows():
        flat = {k: item[k] for k in feed_df.columns}
        prod_id = flat.get("id") or ""
        for _, s in spec_df.iterrows():
            attr = s["Attribute"]
            a_norm = attr.lower()
            requirement = parse_requirement(s.get("Requirement", ""))
            dtype = s.get("Data Type", "")
            supported = s.get("Supported Values", "")
            vrules = s.get("Validation Rules", "")
            if a_norm not in feed_cols_norm:
                if requirement == "Required":
                    rows.append({"record_index": int(idx), "id": prod_id, "attribute": attr, "issue_type": "missing", "details": "Required attribute missing", "value": ""})
                continue
            orig = feed_df.columns[feed_cols_norm.index(a_norm)]
            val = flat.get(orig)
            if val is None or (isinstance(val, str) and not str(val).strip()):
                if requirement == "Required":
                    rows.append({"record_index": int(idx), "id": prod_id, "attribute": attr, "issue_type": "empty", "details": "Required value empty", "value": ""})
                continue
            ok_t, det_t = check_type(pd.Series([val]), dtype)
            if not ok_t:
                rows.append({"record_index": int(idx), "id": prod_id, "attribute": attr, "issue_type": "type", "details": det_t, "value": str(val)})
            ok_v, det_v = check_supported_values(pd.Series([val]), supported)
            if not ok_v:
                rows.append({"record_index": int(idx), "id": prod_id, "attribute": attr, "issue_type": "value", "details": det_v, "value": str(val)})
            ok_r, det_r = apply_validation_rules(pd.Series([val]), vrules)
            if not ok_r:
                rows.append({"record_index": int(idx), "id": prod_id, "attribute": attr, "issue_type": "validation", "details": det_r, "value": str(val)})
    if not rows:
        return pd.DataFrame(columns=["record_index","id","attribute","issue_type","details","value"])
    return pd.DataFrame(rows)

def preview_fuzzy_mapping(feed_df: pd.DataFrame, spec_df: pd.DataFrame):
    spec_cols = spec_df["Attribute"].tolist()
    mapping = []
    for c in list(feed_df.columns):
        match = fuzzy_match_column(c, spec_cols, feed_df)
        mapping.append({"original": c, "mapped_to": match if match else "(no match)"})
    return pd.DataFrame(mapping)
# ============================================================
# PART 3 — STREAMLIT UI, MAIN FLOW, ENRICHMENT & VALIDATION RUN
# ============================================================

st.title("🔎 Auto-Enriched ChatGPT Feed Validator — Smart Mapping")
st.write("""
Upload a JSON or XML product feed. This tool will:
- auto-flatten JSON/XML,
- smart fuzzy-map columns (with title/description detection),
- auto-enrich Shopify-style feeds into ChatGPT-spec fields,
- validate at attribute and per-product level,
- let you inspect products with expandable panels and download CSV reports.
""")

# Sidebar controls
st.sidebar.header("Settings")
max_expand = st.sidebar.number_input("Max Products to Expand in UI", min_value=10, max_value=5000, value=250, step=10)
allow_all = st.sidebar.checkbox("Allow unlimited expanders (may be slow for large feeds)", value=False)
default_currency = st.sidebar.text_input("Default currency to append for prices (ISO 4217)", value="USD")
st.sidebar.markdown("---")
st.sidebar.write("Smart mapping will preferentially map free-text columns to title/description and avoid mapping them to enums.")

uploaded = st.file_uploader("📤 Upload Product Feed (JSON or XML)", type=["json", "xml"])
if not uploaded:
    st.info("Upload a feed file (JSON or XML) to start validation.")
else:
    try:
        uploaded.seek(0)
        ext = uploaded.name.lower().split(".")[-1]

        # Parse & flatten
        with st.spinner("Parsing feed..."):
            if ext == "json":
                feed_df, raw_items = json_to_records_with_raw(uploaded)
            else:
                feed_df = xml_to_records(uploaded)
                raw_items = []

        if feed_df is None or feed_df.empty:
            st.error("Parsing failed — no records found. Check feed structure.")
            st.stop()

        st.success(f"Parsed {len(feed_df)} record(s).")

        # Apply initial fuzzy mapping (smart)
        with st.spinner("Applying smart fuzzy mapping..."):
            feed_df = apply_fuzzy_mapping(feed_df, spec_df)

        # Normalize column names
        feed_df.columns = [str(c).strip() for c in feed_df.columns]

        # Enrichment step for raw JSON (Shopify-like)
        if raw_items:
            st.info("Auto-enrichment: deriving ChatGPT-spec fields from nested JSON (Shopify-style).")
            enriched_rows = []
            for i in range(len(feed_df)):
                raw = raw_items[i] if i < len(raw_items) else {}
                flat = {k: feed_df.iloc[i][k] for k in feed_df.columns}
                enriched = enrich_record_from_shopify(raw, flat, default_currency=default_currency)
                enriched_rows.append(enriched)
            enriched_df = pd.DataFrame(enriched_rows)
            # Re-run fuzzy mapping on enrichment (ensures spec attributes are normalized)
            enriched_df = apply_fuzzy_mapping(enriched_df, spec_df)
            feed_df = enriched_df
        else:
            st.info("No nested raw JSON to enrich (XML or single-object JSON). Skipping enrichment.")

        st.write("---")
        # Attribute-level validation
        st.markdown("## 1️⃣ Attribute-Level Validation")
        with st.spinner("Validating feed attributes..."):
            attr_report = validate_feed_attribute_level(spec_df, feed_df)
        st.dataframe(attr_report, use_container_width=True)
        st.download_button("⬇️ Download Attribute-Level Report (CSV)", attr_report.to_csv(index=False).encode("utf-8"), "attribute_report.csv", "text/csv")

        # Coverage summary
        st.markdown("## 2️⃣ Field Coverage Summary")
        with st.spinner("Computing coverage..."):
            cov = coverage_summary(feed_df, spec_df)
        st.dataframe(cov, use_container_width=True)
        st.download_button("⬇️ Download Field Coverage (CSV)", cov.to_csv(index=False).encode("utf-8"), "field_coverage.csv", "text/csv")

        # Product-level validation (batch)
        st.markdown("## 3️⃣ Product-Level Validation (Batch)")
        with st.spinner("Validating all products..."):
            product_report = validate_all_products(feed_df, spec_df)
        st.success("Product-level validation complete.")
        st.dataframe(product_report.head(200), use_container_width=True)
        st.download_button("⬇️ Download Product-Level Issues (CSV)", product_report.to_csv(index=False).encode("utf-8"), "product_issues.csv", "text/csv")

        # Expandable per-product UI
        st.markdown("## 4️⃣ Inspect Products (Expandable Panels)")
        expand_limit = len(feed_df) if allow_all else min(len(feed_df), max_expand)
        st.info(f"Showing first **{expand_limit} products** (adjust in sidebar).")

        for idx in range(expand_limit):
            row = feed_df.iloc[idx]
            flattened = {k: row[k] for k in feed_df.columns}
            issues = validate_product_record(flattened, spec_df)

            product_id = flattened.get("id") or ""
            summary_parts = []
            if issues["missing"]:
                summary_parts.append("Missing: " + ", ".join(issues["missing"][:6]))
            if issues["empty"]:
                summary_parts.append("Empty: " + ", ".join(issues["empty"][:6]))
            if issues["type_issues"]:
                summary_parts.append(f"Type issues: {len(issues['type_issues'])}")
            if issues["value_issues"]:
                summary_parts.append(f"Value issues: {len(issues['value_issues'])}")
            if issues["validation_issues"]:
                summary_parts.append(f"Rule issues: {len(issues['validation_issues'])}")
            if issues["extras"]:
                summary_parts.append(f"Extras: {len(issues['extras'])}")

            summary = " | ".join(summary_parts) if summary_parts else "No issues"

            header = f"Product #{idx}"
            if product_id:
                header += f" — id: {product_id}"
            header += f" — {summary}"

            with st.expander(header, expanded=False):
                st.markdown("### Issue Summary")
                st.json(issues)
                st.markdown("### Enriched / Flattened Fields")
                st.dataframe(pd.DataFrame([flattened]).T.rename(columns={0: "value"}), use_container_width=True)

        st.success("Inspection complete.")

    except Exception as e:
        st.error(f"Error during processing: {e}")
        st.exception(e)
# ============================================================
# PART 4 — ADVANCED REPORTS, MAPPING PREVIEW, AND FOOTER
# ============================================================

# -------------------------
# Row-level failures (detailed per-row per-attribute)
# -------------------------
def extract_row_level_failures(feed_df: pd.DataFrame, spec_df: pd.DataFrame):
    """
    Returns DataFrame columns:
    record_index, id, attribute, issue_type (missing/empty/type/value/validation), details, value
    """
    rows = []
    feed_cols_norm = [c.lower() for c in feed_df.columns]

    for idx, item in feed_df.iterrows():
        flat = {k: item[k] for k in feed_df.columns}
        prod_id = flat.get("id") or ""
        for _, s in spec_df.iterrows():
            attr = s["Attribute"]
            a_norm = attr.lower()
            requirement = parse_requirement(s.get("Requirement", ""))
            dtype = s.get("Data Type", "")
            supported = s.get("Supported Values", "")
            vrules = s.get("Validation Rules", "")

            # structural missing
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
                continue

            orig_col = feed_df.columns[feed_cols_norm.index(a_norm)]
            val = flat.get(orig_col)

            # empty
            if val is None or (isinstance(val, str) and not str(val).strip()):
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

            # type
            ok_t, det_t = check_type(pd.Series([val]), dtype)
            if not ok_t:
                rows.append({
                    "record_index": int(idx),
                    "id": prod_id,
                    "attribute": attr,
                    "issue_type": "type",
                    "details": det_t,
                    "value": str(val)
                })

            # supported values
            ok_v, det_v = check_supported_values(pd.Series([val]), supported)
            if not ok_v:
                rows.append({
                    "record_index": int(idx),
                    "id": prod_id,
                    "attribute": attr,
                    "issue_type": "value",
                    "details": det_v,
                    "value": str(val)
                })

            # validation rules
            ok_r, det_r = apply_validation_rules(pd.Series([val]), vrules)
            if not ok_r:
                rows.append({
                    "record_index": int(idx),
                    "id": prod_id,
                    "attribute": attr,
                    "issue_type": "validation",
                    "details": det_r,
                    "value": str(val)
                })

    if not rows:
        return pd.DataFrame(columns=["record_index","id","attribute","issue_type","details","value"])
    return pd.DataFrame(rows)


# -------------------------
# Mapping preview helper
# -------------------------
def preview_fuzzy_mapping(feed_df: pd.DataFrame, spec_df: pd.DataFrame):
    spec_cols = spec_df["Attribute"].tolist()
    mapping = []
    for c in list(feed_df.columns):
        match = fuzzy_match_column(c, spec_cols, feed_df)
        mapping.append({"original": c, "mapped_to": match if match else "(no match)"})
    return pd.DataFrame(mapping)


# -------------------------
# UI: Advanced reports + mapping preview
# -------------------------
st.markdown("---")
st.markdown("## 5️⃣ Advanced: Row-level Failures & Mapping Preview")

col1, col2 = st.columns(2)

with col1:
    if st.button("🔎 Generate Row-level Failures (detailed)"):
        with st.spinner("Scanning feed for row-level failures..."):
            failures_df = extract_row_level_failures(feed_df, spec_df)
        if failures_df.empty:
            st.success("No row-level failures found.")
        else:
            st.write(f"Found {len(failures_df)} row-level issues. Showing first 200 rows:")
            st.dataframe(failures_df.head(200), use_container_width=True)
            st.download_button(
                "⬇️ Download full row-level failures CSV",
                failures_df.to_csv(index=False).encode("utf-8"),
                "row_level_failures.csv",
                "text/csv"
            )

with col2:
    if st.button("🔁 Preview Smart Fuzzy Mapping"):
        with st.spinner("Building mapping preview..."):
            mapping_df = preview_fuzzy_mapping(feed_df, spec_df)
        st.write("Original column → Mapped spec field (smart mapping):")
        st.dataframe(mapping_df, use_container_width=True)
        st.download_button(
            "⬇️ Download mapping preview (CSV)",
            mapping_df.to_csv(index=False).encode("utf-8"),
            "mapping_preview.csv",
            "text/csv"
        )

st.markdown("---")
st.info("Notes: Smart mapping attempts to avoid mapping long free-text fields to enumerated fields. Use the mapping preview to verify mappings. If mapping is incorrect, fix upstream source field names and re-upload.")


# -------------------------
# Footer: requirements and tips
# -------------------------
REQ_TXT = """streamlit
pandas
lxml
"""
st.sidebar.markdown("---")
st.sidebar.subheader("requirements.txt")
st.sidebar.code(REQ_TXT)

st.markdown("### Done — Validator Ready")
st.write("""
Tips & next steps:
- To strengthen validation, enable HTTP HEAD checks for `link` and `image_link` (requires network calls).
- To export a ChatGPT-ready JSON feed, I can add an 'Export Enriched Feed' button that outputs a JSON file matching the spec.
- If you want stricter currency checks, I can auto-detect store currency and normalize prices.
""")
# End of PART 4
