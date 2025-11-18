# ============================================================
# PART 1/4 — IMPORTS, SPEC, PARSING, URL EXTRACTION, SMART MAPPING
# ============================================================

import streamlit as st
import pandas as pd
import json
import xml.etree.ElementTree as ET
from io import StringIO, BytesIO
import re
from datetime import datetime
import os

st.set_page_config(page_title="Auto-Enriched ChatGPT Feed Validator (Fixed)", layout="wide")

# ============================================================
# EMBEDDED SPEC (TAB-SEP)
# ============================================================
SPEC_CSV = """Attribute\tData Type\tSupported Values\tDescription\tExample\tRequirement\tDependencies\tValidation Rules
enable_search\tEnum\ttrue, false\tControls whether the product can be surfaced in ChatGPT search results.\tTRUE\tRequired\t—\tLower-case string
enable_checkout\tEnum\ttrue, false\tAllows direct purchase inside ChatGPT.\tTRUE\tRequired\t—\tLower-case string
id\tString (alphanumeric)\t—\tMerchant product ID (unique)\tSKU12345\tRequired\t—\tMax 100 chars; must remain stable over time
gtin\tString (numeric)\tGTIN, UPC, ISBN\tUniversal product identifier\t1.23457E+11\tRecommended\t—\t8–14 digits; no dashes or spaces
mpn\tString (alphanumeric)\t—\tManufacturer part number\tGPT5\tRequired if gtin missing\t—\tMax 70 chars
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
def load_spec_df():
    df = pd.read_csv(StringIO(SPEC_CSV), sep="\t", engine="python")
    df.columns = [c.strip() for c in df.columns]
    return df

spec_df = load_spec_df()

# ============================================================
# JSON & XML flattening helpers
# ============================================================
def _flatten_json(obj, prefix="", out=None):
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            _flatten_json(v, key, out)
    elif isinstance(obj, list):
        # scalar list -> join, else index nested items
        if all(not isinstance(i, (dict, list)) for i in obj):
            out[prefix] = "|".join([str(i) for i in obj if i is not None])
        else:
            for idx, item in enumerate(obj):
                key = f"{prefix}.{idx}" if prefix else str(idx)
                _flatten_json(item, key, out)
    else:
        out[prefix] = obj
    return out

def json_to_records_with_raw(file_bytes: BytesIO):
    file_bytes.seek(0)
    text = file_bytes.read().decode("utf-8", errors="ignore")
    data = json.loads(text)
    records = []
    raw_items = []
    if isinstance(data, list):
        for item in data:
            raw_items.append(item)
            records.append(_flatten_json(item))
        return pd.DataFrame(records), raw_items
    if isinstance(data, dict):
        # try to find the main nested list
        candidate_lists = []
        def walk(d):
            if isinstance(d, dict):
                for v in d.values():
                    walk(v)
            elif isinstance(d, list):
                candidate_lists.append(d)
        walk(data)
        if candidate_lists:
            largest = max(candidate_lists, key=len)
            for item in largest:
                raw_items.append(item)
                records.append(_flatten_json(item))
            return pd.DataFrame(records), raw_items
        raw_items.append(data)
        return pd.DataFrame([_flatten_json(data)]), raw_items
    return pd.DataFrame(), []

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
            out[tag] = vals if len(vals) > 1 else vals[0]
    text = (elem.text or "").strip()
    if text and not children and not elem.attrib:
        return text
    elif text:
        out["_text"] = text
    return out

def xml_to_records(file_bytes: BytesIO):
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
# SMART URL / IMAGE EXTRACTION
# ============================================================
url_re = re.compile(r"https?://[^\s\"']+", re.IGNORECASE)

def extract_urls_from_any(x):
    """
    Recursively extract URL-like strings from scalars, lists, dicts.
    Returns unique URLs preserving order.
    """
    urls = []
    if x is None:
        return urls
    if isinstance(x, str):
        found = url_re.findall(x)
        for u in found:
            if u not in urls:
                urls.append(u)
        return urls
    if isinstance(x, list):
        for it in x:
            for u in extract_urls_from_any(it):
                if u not in urls:
                    urls.append(u)
        return urls
    if isinstance(x, dict):
        # look for common URL keys first
        for key in ["src", "url", "href", "image", "link"]:
            if key in x and isinstance(x[key], str):
                if url_re.search(x[key]) and x[key] not in urls:
                    urls.append(x[key])
        for v in x.values():
            for u in extract_urls_from_any(v):
                if u not in urls:
                    urls.append(u)
        return urls
    return urls

# ============================================================
# SAFE EMPTY CHECK (prevents Series truth ambiguity)
# ============================================================
def is_empty(val):
    """
    Return True if val is None, NaN, empty str, empty list/tuple, empty Series or all-empty Series.
    Avoids ambiguous truth-value errors from pandas.Series.
    """
    if val is None:
        return True
    if isinstance(val, float) and pd.isna(val):
        return True
    if isinstance(val, (list, tuple)) and len(val) == 0:
        return True
    if isinstance(val, pd.Series):
        # consider empty if all NA or all blank strings
        try:
            if val.isna().all():
                return True
            s = val.astype(str).str.strip()
            return (s == "").all()
        except Exception:
            return False
    if isinstance(val, str) and val.strip() == "":
        return True
    return False

# ============================================================
# SMART FUZZY MAPPER (title vs description detection + protections)
# ============================================================
def canonicalize(col: str) -> str:
    c = str(col).lower().strip()
    c = re.sub(r"[^a-z0-9]+", "", c)
    # strip common prefixes/suffixes lightly
    c = re.sub(r"^(product|item|prod|p|merchant|seller|offer|sku)","", c)
    c = re.sub(r"(code|identifier|id|val|value)$","", c)
    c = re.sub(r"\d+$", "", c)
    return c

def fuzzy_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    prefix_len = len(os.path.commonprefix([a,b]))
    lp = prefix_len / max(1, max(len(a), len(b)))
    set_a, set_b = set(a), set(b)
    overlap = len(set_a & set_b) / max(1, len(set_a | set_b))
    def lcs(s1,s2):
        m = [[0]*(1+len(s2)) for _ in range(1+len(s1))]
        longest = 0
        for i in range(1,1+len(s1)):
            for j in range(1,1+len(s2)):
                if s1[i-1] == s2[j-1]:
                    m[i][j] = m[i-1][j-1] + 1
                    longest = max(longest, m[i][j])
        return longest
    lcs_val = lcs(a,b) / max(1, min(len(a), len(b)))
    return (0.45 * lp) + (0.35 * overlap) + (0.20 * lcs_val)

def detect_free_text_column(series: pd.Series):
    """
    Heuristic to determine title-like vs description-like.
    Returns (is_title_like, is_description_like)
    """
    if series is None or len(series.dropna()) < 5:
        return False, False
    vals = series.dropna().astype(str).head(200).tolist()
    avg_len = sum(len(v) for v in vals) / len(vals)
    distinct_ratio = len(set(vals)) / len(vals)
    is_description_like = avg_len > 60 or distinct_ratio > 0.85
    is_title_like = (10 < avg_len <= 60) and distinct_ratio > 0.5
    return is_title_like, is_description_like

def fuzzy_match_column(col: str, target_fields: list, feed_df: pd.DataFrame=None) -> str | None:
    """
    Smart fuzzy match:
    - If column name explicitly contains 'title' or 'description', favor mapping.
    - If feed_df provided, detect free-text and prefer title/description mapping.
    - Avoid mapping free-text to enumerated fields.
    - Use a higher threshold (0.55) for fuzzy matches.
    """
    canon = canonicalize(col)
    is_title_like = False
    is_description_like = False
    if feed_df is not None and col in feed_df.columns:
        try:
            is_title_like, is_description_like = detect_free_text_column(feed_df[col])
        except Exception:
            is_title_like, is_description_like = False, False

    # explicit name hints
    low = col.lower()
    if "title" in low and not is_description_like:
        return "title"
    if "description" in low or "body_html" in low:
        return "description"

    if is_description_like:
        return "description"
    if is_title_like:
        return "title"

    # avoid mapping titles/description to enum targets
    enum_fields = {"condition","gender","age_group","availability","pickup_method","relationship_type"}

    # exact canonical match
    for t in target_fields:
        if canonicalize(t) == canon:
            return t

    # token overlap quick pass
    tokens = re.split(r"[_\-\s\.]+", col.lower())
    for t in target_fields:
        if any(tok in t.lower() for tok in tokens if len(tok) > 2):
            # don't map to enum if this looks like free-text
            if t.lower() in enum_fields and (is_title_like or is_description_like):
                continue
            return t

    # fuzzy scoring
    best = None
    best_score = 0.0
    for t in target_fields:
        if t.lower() in enum_fields and (is_title_like or is_description_like):
            continue
        score = fuzzy_score(canon, canonicalize(t))
        if score > best_score:
            best_score = score
            best = t
    if best_score >= 0.55:
        return best
    return None

def apply_fuzzy_mapping(feed_df: pd.DataFrame, spec_df: pd.DataFrame) -> pd.DataFrame:
    spec_cols = spec_df["Attribute"].tolist()
    mapping = {}
    used_targets = set()
    for c in list(feed_df.columns):
        match = fuzzy_match_column(c, spec_cols, feed_df)
        if match and match not in used_targets:
            mapping[c] = match
            used_targets.add(match)
        elif match and match in used_targets:
            mapping[c] = f"{match}__alt"
        else:
            mapping[c] = c
    return feed_df.rename(columns=mapping)

# End of PART 1/4
# ============================================================
# PART 2/4 — VALIDATION HELPERS, ENRICHMENT, PER-PRODUCT VALIDATION
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
    Validate values in a Series against a dtype hint.
    Series-safe (never uses truthiness checks).
    """
    if dtype_hint is None or pd.isna(dtype_hint):
        return True, ""

    s = str(dtype_hint).lower()
    non_null = series.dropna().astype(str)

    if non_null.empty:
        return True, ""

    # Integer
    if "integer" in s or ("int" in s and "float" not in s):
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
            for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y",
                        "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S"):
                try:
                    datetime.strptime(x, fmt)
                    return True
                except:
                    pass
            return False

        ok = non_null.apply(try_parse)
        if not ok.all():
            return False, "Contains invalid date values"
        return True, ""

    # URL
    if "url" in s:
        ok = non_null.apply(
            lambda x: bool(re.search(r"^https?://", x.lower()))
        )
        if not ok.all():
            return False, "Invalid URL format"
        return True, ""

    return True, ""


# -------------------------
# Supported values checker
# -------------------------
def check_supported_values(series: pd.Series, supported_str: str):
    if pd.isna(supported_str) or not supported_str:
        return True, ""

    allowed = [
        a.strip().lower() for a in re.split(r"[,|;]", str(supported_str))
        if a.strip()
    ]
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

    rules = [r.strip() for r in str(rules_text).split(";") if r.strip()]
    details = []
    s = series.dropna().astype(str)

    for rule in rules:
        rl = rule.lower()

        # Max length
        if "max" in rl:
            m = re.search(r"max.*?(\d+)", rl)
            if m:
                mx = int(m.group(1))
                too_long = s.apply(len) > mx
                if too_long.any():
                    details.append(f"{too_long.sum()} values exceed max length {mx}")

        # Min length
        if "min" in rl:
            m = re.search(r"min.*?(\d+)", rl)
            if m:
                mn = int(m.group(1))
                too_short = s.apply(len) < mn
                if too_short.any():
                    details.append(f"{too_short.sum()} values shorter than min length {mn}")

        # Regex
        if "regex:" in rl:
            pattern = rule.split("regex:", 1)[1]
            try:
                pat = re.compile(pattern)
                bad = ~s.apply(lambda x: bool(pat.search(x)))
                if bad.any():
                    details.append(f"{bad.sum()} values fail regex")
            except:
                details.append("Invalid regex")

        # Unique
        if "unique" in rl:
            dup = s.duplicated().sum()
            if dup > 0:
                details.append(f"{dup} duplicate values")

    ok = len(details) == 0
    return ok, "; ".join(details)


# -------------------------
# HTML stripping
# -------------------------
def strip_html_tags(text):
    if not isinstance(text, str):
        return text
    clean = re.sub(r"<[^>]+>", "", text)
    return clean.strip()


# ============================================================
# SHOPIFY / GENERIC FEED AUTO-ENRICHMENT (Option B)
# ============================================================

def enrich_record_from_shopify(raw_item: dict, flat_record: dict, default_currency="USD"):
    """
    Produces the correct ChatGPT feed fields.
    Fixed: uses is_empty() everywhere so Series/lists cannot break logic.
    """

    enriched = dict(flat_record)  # base fields

    # ----------------------------
    # TITLE
    # ----------------------------
    if is_empty(enriched.get("title")):
        if isinstance(raw_item.get("title"), str):
            enriched["title"] = raw_item["title"]

    # ----------------------------
    # DESCRIPTION
    # ----------------------------
    if is_empty(enriched.get("description")):
        bh = (
            raw_item.get("body_html")
            or raw_item.get("description")
            or raw_item.get("html_description")
        )
        if isinstance(bh, str):
            enriched["description"] = strip_html_tags(bh)

    # ----------------------------
    # BRAND
    # ----------------------------
    if is_empty(enriched.get("brand")):
        if isinstance(raw_item.get("vendor"), str):
            enriched["brand"] = raw_item["vendor"]

    # ----------------------------
    # CATEGORY
    # ----------------------------
    if is_empty(enriched.get("product_category")):
        if isinstance(raw_item.get("product_type"), str):
            enriched["product_category"] = raw_item["product_type"]
        else:
            tags = raw_item.get("tags")
            if isinstance(tags, list) and len(tags):
                enriched["product_category"] = ", ".join(tags[:3])
            elif isinstance(tags, str):
                enriched["product_category"] = tags

    # ----------------------------
    # IMAGES / MEDIA
    # ----------------------------
    images = []

    # from Shopify 'images'
    if isinstance(raw_item.get("images"), list):
        for img in raw_item["images"]:
            urls = extract_urls_from_any(img)
            for u in urls:
                if u not in images:
                    images.append(u)

    # also try media/photos/gallery/assets
    for group in ["media", "photos", "gallery", "assets"]:
        if isinstance(raw_item.get(group), list):
            for val in raw_item[group]:
                urls = extract_urls_from_any(val)
                for u in urls:
                    if u not in images:
                        images.append(u)

    # also check flattened fields: might include stray URLs
    for v in flat_record.values():
        if isinstance(v, str):
            for u in url_re.findall(v):
                if u not in images:
                    images.append(u)

    if not is_empty(enriched.get("image_link")) and isinstance(enriched.get("image_link"), str):
        # already enriched via fuzzy mapping
        pass
    else:
        if images:
            enriched["image_link"] = images[0]

    # Additional images
    if is_empty(enriched.get("additional_image_link")):
        if len(images) > 1:
            enriched["additional_image_link"] = ", ".join(images[1:])
        else:
            enriched["additional_image_link"] = ""

    # ----------------------------
    # HANDLE VARIANTS
    # ----------------------------
    variants = raw_item.get("variants") or []
    if isinstance(variants, list) and len(variants):
        first = variants[0]

        # PRICE
        if is_empty(enriched.get("price")):
            p = (
                first.get("price")
                or first.get("compare_at_price")
                or first.get("presentment_price")
            )
            if p is not None:
                p = str(p)
                if re.search(r"[A-Z]{3}$", p):  # already OK
                    enriched["price"] = p
                else:
                    enriched["price"] = f"{p} {default_currency}"

        # SALE PRICE
        if is_empty(enriched.get("sale_price")):
            cp = first.get("compare_at_price") or first.get("compare_price")
            if cp:
                cp = str(cp)
                if re.search(r"[A-Z]{3}$", cp):
                    enriched["sale_price"] = cp
                else:
                    enriched["sale_price"] = f"{cp} {default_currency}"

        # MPN from SKU
        if is_empty(enriched.get("mpn")):
            sku = first.get("sku") or first.get("mpn")
            if sku:
                enriched["mpn"] = str(sku)

        # Availability & Inventory
        any_avail = False
        total_inv = 0
        inv_found = False

        for v in variants:
            # availability
            if v.get("available") in (True, "True", "true", 1, "1"):
                any_avail = True

            # inventory
            if v.get("inventory_quantity") not in (None, ""):
                try:
                    total_inv += int(v.get("inventory_quantity"))
                    inv_found = True
                except:
                    pass

        if is_empty(enriched.get("availability")):
            enriched["availability"] = "in_stock" if any_avail else "out_of_stock"

        if is_empty(enriched.get("inventory_quantity")) and inv_found:
            enriched["inventory_quantity"] = total_inv

    # ----------------------------
    # DETERMINE PRODUCT LINK
    # ----------------------------
    if is_empty(enriched.get("link")):
        for k in ["url", "product_url", "product_permalink", "permalink", "page_url"]:
            if isinstance(raw_item.get(k), str) and url_re.search(raw_item[k]):
                enriched["link"] = raw_item[k]
                break

    # fallback: Shopify "handle" into URL if base domain known
    if is_empty(enriched.get("link")) and isinstance(raw_item.get("handle"), str):
        # no domain known — leave blank instead of guessing
        pass

    # ----------------------------
    # ID fallback
    # ----------------------------
    if is_empty(enriched.get("id")):
        if raw_item.get("id") is not None:
            enriched["id"] = raw_item.get("id")

    return enriched


# ============================================================
# PER-PRODUCT VALIDATION
# ============================================================

def validate_product_record(record: dict, spec_df: pd.DataFrame):
    """
    Validate a single enriched JSON/XML record.
    Returns dict with missing / empty / type / value / rule / extras.
    """
    issues = {
        "missing": [],
        "empty": [],
        "type_issues": [],
        "value_issues": [],
        "validation_issues": [],
        "extras": []
    }

    record_keys = {k.lower(): k for k in record.keys()}
    spec_attrs = spec_df["Attribute"].tolist()
    spec_norm = [a.lower() for a in spec_attrs]

    # mark extras
    for k in record.keys():
        if k.lower() not in spec_norm:
            issues["extras"].append(k)

    # validate spec fields
    for _, row in spec_df.iterrows():
        attr = row["Attribute"]
        a_norm = attr.lower()
        dtype = row["Data Type"]
        supported = row["Supported Values"]
        vrules = row["Validation Rules"]
        req = parse_requirement(row["Requirement"])

        # missing
        if a_norm not in record_keys:
            if req == "Required":
                issues["missing"].append(attr)
            continue

        orig_key = record_keys[a_norm]
        val = record.get(orig_key)

        # empty
        if is_empty(val):
            if req == "Required":
                issues["empty"].append(attr)
            continue

        # Validate type
        ok_t, tdet = check_type(pd.Series([val]), dtype)
        if not ok_t:
            issues["type_issues"].append(f"{attr}: {tdet}")

        # Validate allowed values
        ok_v, vdet = check_supported_values(pd.Series([val]), supported)
        if not ok_v:
            issues["value_issues"].append(f"{attr}: {vdet}")

        # Validate rules
        ok_r, rdet = apply_validation_rules(pd.Series([val]), vrules)
        if not ok_r:
            issues["validation_issues"].append(f"{attr}: {rdet}")

    return issues


# ============================================================
# FEED-LEVEL VALIDATION (for coverage table + attribute summary)
# ============================================================

def coverage_summary(feed_df: pd.DataFrame, spec_df: pd.DataFrame):
    rows = []
    feed_cols = list(feed_df.columns)
    feed_norm = [c.lower() for c in feed_cols]

    for _, s in spec_df.iterrows():
        attr = s["Attribute"]
        a_norm = attr.lower()
        if a_norm in feed_norm:
            col = feed_cols[feed_norm.index(a_norm)]
            filled = 100 - feed_df[col].isna().mean() * 100
            example = next((str(v) for v in feed_df[col] if not is_empty(v)), "")
            rows.append({
                "Attribute": attr,
                "Present": True,
                "% Filled": f"{filled:.1f}%",
                "Example": example
            })
        else:
            rows.append({
                "Attribute": attr,
                "Present": False,
                "% Filled": "0%",
                "Example": ""
            })

    extras = [c for c in feed_cols if c.lower() not in [a.lower() for a in spec_df["Attribute"]]]
    if extras:
        rows.append({
            "Attribute": "(Extra fields)",
            "Present": True,
            "% Filled": "",
            "Example": ", ".join(extras[:10])
        })

    return pd.DataFrame(rows)


def validate_feed_attribute_level(spec_df: pd.DataFrame, feed_df: pd.DataFrame):
    spec_norm = spec_df["Attribute"].str.lower().tolist()
    feed_cols = list(feed_df.columns)
    feed_norm = [c.lower() for c in feed_cols]

    report = []

    for _, s in spec_df.iterrows():
        attr = s["Attribute"]
        a_norm = attr.lower()
        dtype = s.get("Data Type", "")
        supported = s.get("Supported Values", "")
        vrules = s.get("Validation Rules", "")
        req = parse_requirement(s.get("Requirement", ""))

        if a_norm not in feed_norm:
            status = (
                "❌ Missing (Required)" if req == "Required"
                else ("⚠️ Missing (Recommended)" if req == "Recommended" else "ℹ️ Missing (Optional)")
            )
            report.append({
                "Attribute": attr,
                "Requirement": req,
                "Exists": False,
                "Status": status,
                "Details": "Not present"
            })
            continue

        orig_col = feed_cols[feed_norm.index(a_norm)]
        col = feed_df[orig_col]
        details = []

        empty_pct = col.apply(is_empty).mean() * 100
        if empty_pct > 0:
            details.append(f"{empty_pct:.1f}% empty")

        ok_t, tdet = check_type(col, dtype)
        if not ok_t:
            details.append(tdet)

        ok_v, vdet = check_supported_values(col, supported)
        if not ok_v:
            details.append(vdet)

        ok_r, rdet = apply_validation_rules(col, vrules)
        if not ok_r:
            details.append(rdet)

        status = "✅ Valid" if not details else "⚠️ Issues"
        report.append({
            "Attribute": attr,
            "Requirement": req,
            "Exists": True,
            "Status": status,
            "Details": " | ".join(details)
        })

    return pd.DataFrame(report)


# ============================================================
# BATCH PRODUCT VALIDATION
# ============================================================

def validate_all_products(feed_df: pd.DataFrame, spec_df: pd.DataFrame):
    rows = []

    for idx, row in feed_df.iterrows():
        record = {k: row[k] for k in feed_df.columns}
        issues = validate_product_record(record, spec_df)
        pid = record.get("id", "")

        rows.append({
            "record_index": idx,
            "id": pid,
            "missing": "|".join(issues["missing"]),
            "empty": "|".join(issues["empty"]),
            "type_issues": "|".join(issues["type_issues"]),
            "value_issues": "|".join(issues["value_issues"]),
            "validation_issues": "|".join(issues["validation_issues"]),
            "extras": "|".join(issues["extras"]),
            "has_issues": any(len(x) > 0 for x in issues.values())
        })

    return pd.DataFrame(rows)

# End of PART 2/4
# ============================================================
# PART 3/4 — STREAMLIT UI, MAIN PIPELINE, ENRICHMENT & VALIDATION
# ============================================================

st.title("🔎 Auto-Enriched ChatGPT Product Feed Validator")
st.caption("Smart Fuzzy Mapping • Shopify-Style Enrichment • Per-Product Issues • Arrays/Images Supported")

st.markdown("""
This tool automatically:
- Detects JSON or XML schema  
- Flattens + extracts nested URLs/images  
- Performs *smart* fuzzy column mapping  
- Auto-enriches Shopify-style raw data (Option B)  
- Validates feed against the **full ChatGPT Commerce Product Spec**  
- Produces attribute-level and product-level reports  
- Shows expandable product-level panels for debugging  
""")

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Options")

max_expand = st.sidebar.number_input(
    "Max Expandable Products",
    min_value=10,
    max_value=5000,
    value=200,
    step=10
)
allow_all = st.sidebar.checkbox("Allow unlimited product expanders (slower for large feeds)", value=False)

default_currency = st.sidebar.text_input(
    "Default Currency (ISO 4217)",
    value="USD"
)

uploaded_file = st.file_uploader("📤 Upload your JSON or XML product feed", type=["json", "xml"])

if not uploaded_file:
    st.info("Upload a JSON or XML feed to begin.")
    st.stop()

# ============================================================
# PROCESS FILE
# ============================================================
try:
    ext = uploaded_file.name.lower().split(".")[-1]

    with st.spinner("Parsing feed..."):
        if ext == "json":
            feed_df, raw_items = json_to_records_with_raw(uploaded_file)
        else:
            raw_items = []
            feed_df = xml_to_records(uploaded_file)

    if feed_df is None or feed_df.empty:
        st.error("No valid product records found in file.")
        st.stop()

    st.success(f"Parsed {len(feed_df)} products.")

    # ============================================================
    # SMART FUZZY MAPPING ON FLATTENED FIELDS
    # ============================================================
    with st.spinner("Applying smart fuzzy mapping..."):
        feed_df = apply_fuzzy_mapping(feed_df, spec_df)

    feed_df.columns = [c.strip() for c in feed_df.columns]

    # ============================================================
    # AUTO-ENRICH USING RAW JSON (Shopify-like)
    # ============================================================
    if raw_items:
        st.info("Auto-enrichment enabled: Shopify-style raw fields detected.")

        enriched_records = []
        for i in range(len(feed_df)):
            raw = raw_items[i] if i < len(raw_items) else {}
            flat = {k: feed_df.iloc[i][k] for k in feed_df.columns}

            enriched = enrich_record_from_shopify(
                raw,
                flat,
                default_currency=default_currency
            )
            enriched_records.append(enriched)

        enriched_df = pd.DataFrame(enriched_records)

        # Re-run fuzzy mapping after enrichment
        enriched_df = apply_fuzzy_mapping(enriched_df, spec_df)
        feed_df = enriched_df

    else:
        st.info("No raw JSON objects found — auto-enrichment skipped.")

    # ============================================================
    # ATTRIBUTE-LEVEL VALIDATION
    # ============================================================
    st.markdown("## 1️⃣ Attribute-Level Validation")

    with st.spinner("Validating attribute coverage..."):
        attr_report = validate_feed_attribute_level(spec_df, feed_df)

    st.dataframe(attr_report, use_container_width=True)
    st.download_button(
        "⬇️ Download Attribute Report (CSV)",
        attr_report.to_csv(index=False).encode("utf-8"),
        "attribute_report.csv",
        "text/csv"
    )

    # ============================================================
    # FIELD COVERAGE SUMMARY
    # ============================================================
    st.markdown("## 2️⃣ Field Coverage Summary")

    with st.spinner("Computing coverage..."):
        cov_df = coverage_summary(feed_df, spec_df)

    st.dataframe(cov_df, use_container_width=True)
    st.download_button(
        "⬇️ Download Coverage Summary (CSV)",
        cov_df.to_csv(index=False).encode("utf-8"),
        "coverage_summary.csv",
        "text/csv"
    )

    # ============================================================
    # PRODUCT-LEVEL VALIDATION
    # ============================================================
    st.markdown("## 3️⃣ Product-Level Validation")

    with st.spinner("Validating all products..."):
        product_report = validate_all_products(feed_df, spec_df)

    st.dataframe(product_report.head(200), use_container_width=True)

    st.download_button(
        "⬇️ Download Product-Level Issues (CSV)",
        product_report.to_csv(index=False).encode("utf-8"),
        "product_issues.csv",
        "text/csv"
    )

    # ============================================================
    # EXPANDABLE PER-PRODUCT DEBUGGING
    # ============================================================
    st.markdown("## 4️⃣ Inspect Products (Expandable Panels)")

    limit = len(feed_df) if allow_all else min(max_expand, len(feed_df))
    st.info(f"Showing **{limit} products**.")

    for idx in range(limit):
        row = feed_df.iloc[idx]
        record = {k: row[k] for k in feed_df.columns}
        issues = validate_product_record(record, spec_df)

        pid = record.get("id", "")
        summary_parts = []

        if issues["missing"]:
            summary_parts.append("Missing: " + ", ".join(issues["missing"][:5]))
        if issues["empty"]:
            summary_parts.append("Empty: " + ", ".join(issues["empty"][:5]))
        if issues["type_issues"]:
            summary_parts.append("Type: " + str(len(issues["type_issues"])))
        if issues["value_issues"]:
            summary_parts.append("Value: " + str(len(issues["value_issues"])))
        if issues["validation_issues"]:
            summary_parts.append("Rules: " + str(len(issues["validation_issues"])))
        if issues["extras"]:
            summary_parts.append("Extras: " + str(len(issues["extras"])))

        summary = " | ".join(summary_parts) if summary_parts else "No issues"

        header = f"Product #{idx}"
        if pid:
            header += f" — id: {pid}"
        header += f" — {summary}"

        with st.expander(header):
            st.markdown("### Issue Breakdown")
            st.json(issues)

            st.markdown("### Product Fields")
            st.dataframe(
                pd.DataFrame([record]).T.rename(columns={0: "value"}),
                use_container_width=True
            )

    st.success("Inspection complete.")

except Exception as e:
    st.error(f"Error during processing: {e}")
    st.exception(e)
# ============================================================
# PART 4/4 — ADVANCED REPORTS, MAPPING PREVIEW, FOOTER
# ============================================================

# ------------------------------------------------------------
# ROW-LEVEL FAILURE EXTRACTION (detailed per row/attribute)
# ------------------------------------------------------------
def extract_row_level_failures(feed_df: pd.DataFrame, spec_df: pd.DataFrame):
    rows = []
    feed_cols_norm = [c.lower() for c in feed_df.columns]

    for idx, item in feed_df.iterrows():
        rec = {k: item[k] for k in feed_df.columns}
        pid = rec.get("id", "")
        for _, s in spec_df.iterrows():
            attr = s["Attribute"]
            a_norm = attr.lower()
            dtype = s["Data Type"]
            supported = s["Supported Values"]
            vrules = s["Validation Rules"]
            req = parse_requirement(s["Requirement"])

            # Missing field
            if a_norm not in feed_cols_norm:
                if req == "Required":
                    rows.append({
                        "record_index": idx,
                        "id": pid,
                        "attribute": attr,
                        "issue_type": "missing",
                        "details": "Required attribute not present in feed",
                        "value": ""
                    })
                continue

            orig_col = feed_df.columns[feed_cols_norm.index(a_norm)]
            val = rec.get(orig_col)

            # Empty
            if is_empty(val):
                if req == "Required":
                    rows.append({
                        "record_index": idx,
                        "id": pid,
                        "attribute": attr,
                        "issue_type": "empty",
                        "details": "Required value is empty",
                        "value": ""
                    })
                continue

            # Type
            ok_t, det_t = check_type(pd.Series([val]), dtype)
            if not ok_t:
                rows.append({
                    "record_index": idx,
                    "id": pid,
                    "attribute": attr,
                    "issue_type": "type",
                    "details": det_t,
                    "value": str(val)
                })

            # Supported value sets
            ok_v, det_v = check_supported_values(pd.Series([val]), supported)
            if not ok_v:
                rows.append({
                    "record_index": idx,
                    "id": pid,
                    "attribute": attr,
                    "issue_type": "value",
                    "details": det_v,
                    "value": str(val)
                })

            # Validation rules
            ok_r, det_r = apply_validation_rules(pd.Series([val]), vrules)
            if not ok_r:
                rows.append({
                    "record_index": idx,
                    "id": pid,
                    "attribute": attr,
                    "issue_type": "validation",
                    "details": det_r,
                    "value": str(val)
                })

    if not rows:
        return pd.DataFrame(columns=[
            "record_index", "id", "attribute",
            "issue_type", "details", "value"
        ])

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# FUZZY MAPPING PREVIEW
# ------------------------------------------------------------
def preview_fuzzy_mapping(feed_df: pd.DataFrame, spec_df: pd.DataFrame):
    spec_cols = spec_df["Attribute"].tolist()
    rows = []
    for c in feed_df.columns:
        match = fuzzy_match_column(c, spec_cols, feed_df)
        rows.append({
            "original_column": c,
            "mapped_to": match if match else "(no match)"
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# ADVANCED REPORT SECTION
# ------------------------------------------------------------

st.markdown("---")
st.markdown("## 5️⃣ Advanced Diagnostics")

colA, colB = st.columns(2)

# Row-level failure analysis
with colA:
    if st.button("🔍 Generate Row-Level Failures"):
        with st.spinner("Scanning each record and attribute..."):
            failure_df = extract_row_level_failures(feed_df, spec_df)

        if failure_df.empty:
            st.success("No detailed row-level failures found.")
        else:
            st.warning(f"Found {len(failure_df)} issues.")
            st.dataframe(failure_df.head(200), use_container_width=True)

            st.download_button(
                "⬇️ Download Full Row-Level Failures (CSV)",
                failure_df.to_csv(index=False).encode("utf-8"),
                "row_level_failures.csv",
                "text/csv"
            )

# Smart fuzzy mapping preview
with colB:
    if st.button("🔁 Preview Fuzzy Mapping"):
        with st.spinner("Building mapping preview..."):
            mapping_df = preview_fuzzy_mapping(feed_df, spec_df)

        st.dataframe(mapping_df, use_container_width=True)

        st.download_button(
            "⬇️ Download Mapping Preview (CSV)",
            mapping_df.to_csv(index=False).encode("utf-8"),
            "mapping_preview.csv",
            "text/csv"
        )

# ------------------------------------------------------------
# REQUIREMENTS.TXT DISPLAY
# ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("requirements.txt")
st.sidebar.code("streamlit\npandas\nlxml\n")

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("---")
st.info("""
### 🔧 You're all set!

This validator:
- Smart-maps columns using title/description heuristics  
- Extracts URLs from arrays, lists, nested objects  
- Auto-enriches Shopify-style feeds  
- Performs full ChatGPT Product Spec validation  
- Generates attribute-level, product-level, and full row-level reports  

If you want additional features such as:
- Auto-export into ChatGPT-ready JSON  
- Parent/variant folding  
- Strict URL 200-status checking  
- Automatic variant explosion or collapse  
Just tell me — I can add them.
""")

# End of PART 4/4
