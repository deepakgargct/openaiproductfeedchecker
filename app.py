# app.py — Auto-enriched Feed Builder + Fuzzy Mapping + Product-Level Validator
import streamlit as st
import pandas as pd
import json
import xml.etree.ElementTree as ET
from io import StringIO, BytesIO
import re
from datetime import datetime
import os

st.set_page_config(page_title="Auto Schema Feed Validator — Enriched", layout="wide")

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
material\tString\t—\tPrimary material(s\tLeather\tRequired\t—\tMax 100 chars
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

# new: parse JSON and return list of raw items + flattened df
def json_to_records_with_raw(file_bytes: BytesIO):
    file_bytes.seek(0)
    data_text = file_bytes.read().decode("utf-8", errors="ignore")
    data = json.loads(data_text)
    records = []
    raw_items = []
    if isinstance(data, list):
        for item in data:
            raw_items.append(item)
            records.append(_flatten_json(item))
        return pd.DataFrame(records), raw_items
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
                raw_items.append(item)
                records.append(_flatten_json(item))
            return pd.DataFrame(records), raw_items
        # single object
        raw_items.append(data)
        return pd.DataFrame([_flatten_json(data)]), raw_items
    return pd.DataFrame(), []

# -------------------------
# XML parsing -> records (unchanged)
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
# URL extractor for arrays (NEW)
# -------------------------
url_re_global = re.compile(r"https?://[^\s\"']+", re.IGNORECASE)

def extract_urls_from_any(x):
    """
    Recursively extract URL-like strings from scalars, dicts, lists.
    Returns a list of unique URLs in order.
    """
    urls = []
    if x is None:
        return urls
    if isinstance(x, str):
        found = url_re_global.findall(x)
        return found
    if isinstance(x, list):
        for item in x:
            for u in extract_urls_from_any(item):
                if u not in urls:
                    urls.append(u)
        return urls
    if isinstance(x, dict):
        # prioritize common keys
        for key in ["src","url","image","href","link"]:
            if key in x and isinstance(x[key], str) and url_re_global.search(x[key]):
                if x[key] not in urls:
                    urls.append(x[key])
        # then scan all values
        for v in x.values():
            for u in extract_urls_from_any(v):
                if u not in urls:
                    urls.append(u)
        return urls
    return urls

# -------------------------
# FUZZY HEADER MAPPER (unchanged but included)
# -------------------------
def canonicalize(col: str) -> str:
    col = str(col).lower().strip()
    col = re.sub(r"[^a-z0-9]+", "", col)
    col = re.sub(r"^(product|productid|item|prod|p|merchant|seller|offer|sku|sku_code|skuid)","", col)
    col = re.sub(r"(code|identifier|identifierid|val|value)$","", col)
    col = re.sub(r"\d+$", "", col)
    return col

def fuzzy_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    prefix_len = len(os.path.commonprefix([a, b]))
    lp = prefix_len / max(len(a), len(b))
    set_a = set(a)
    set_b = set(b)
    overlap = len(set_a & set_b) / max(1, len(set_a | set_b))
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
    score = (0.45 * lp) + (0.35 * overlap) + (0.20 * lcs)
    return score

def fuzzy_match_column(col: str, target_list: list) -> str or None:
    canon = canonicalize(col)
    for t in target_list:
        if canonicalize(t) == canon:
            return t
    tokens = re.split(r"[_\-\s\.]+", str(col).lower())
    for t in target_list:
        tkns = re.split(r"[_\-\s\.]+", str(t).lower())
        if any(tok in t.lower() for tok in tokens if len(tok) > 2):
            return t
    best = None
    best_score = 0.0
    for t in target_list:
        tc = canonicalize(t)
        score = fuzzy_score(canon, tc)
        if score > best_score:
            best_score = score
            best = t
    if best_score >= 0.45:
        return best
    return None

def apply_fuzzy_mapping(feed_df: pd.DataFrame, spec_df: pd.DataFrame) -> pd.DataFrame:
    spec_cols = spec_df['Attribute'].tolist()
    mapping = {}
    used_targets = set()
    for c in list(feed_df.columns):
        match = fuzzy_match_column(c, spec_cols)
        if match and match not in used_targets:
            mapping[c] = match
            used_targets.add(match)
        elif match and match in used_targets:
            mapping[c] = f"{match}__alt"
        else:
            mapping[c] = c
    new_df = feed_df.rename(columns=mapping)
    return new_df

# -------------------------
# Spec parsing helpers (requirement, type checks, rules) unchanged
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
    if dtype_str is None or pd.isna(dtype_str) or str(dtype_str).strip() == "":
        return True, ""
    s = str(dtype_str).lower()
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return True, ""
    if ("int" in s and "float" not in s) or ("integer" in s):
        coerced = pd.to_numeric(non_null, errors="coerce")
        if coerced.isna().any():
            return False, "Contains non-integer values"
        return True, ""
    if ("number" in s) or ("float" in s) or ("price" in s) or ("percentage" in s):
        cleaned = non_null.str.replace(r"[^\d\.\-]", "", regex=True)
        coerced = pd.to_numeric(cleaned, errors="coerce")
        if coerced.isna().any():
            return False, "Contains non-numeric values"
        return True, ""
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
    if "url" in s:
        pattern = re.compile(r"^https?://", re.I)
        ok = non_null.apply(lambda x: bool(pattern.search(x)))
        if not ok.all():
            return False, "Some values do not start with http(s)://"
        return True, ""
    return True, ""

def check_supported_values(series: pd.Series, supported_str: str):
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

def apply_validation_rules(series: pd.Series, rules_text: str):
    if pd.isna(rules_text) or str(rules_text).strip() == "":
        return True, ""
    parts = [p.strip() for p in str(rules_text).split(";") if p.strip() != ""]
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
        if low.startswith("regex:") or "regex:" in low:
            pattern = p.split("regex:", 1)[1]
            try:
                pat = re.compile(pattern)
                bad = ~s.apply(lambda x: bool(pat.search(x)))
                if bad.any():
                    details.append(f"{bad.sum()} values do not match regex")
            except Exception as e:
                details.append(f"Invalid regex: {e}")
        if "unique" in low:
            if "yes" in low or "true" in low or ":yes" in low:
                dup = s.duplicated().sum()
                if dup > 0:
                    details.append(f"{dup} duplicate values (should be unique)")
    ok = len(details) == 0
    return ok, "; ".join(details)

# -------------------------
# Auto-enrichment for Shopify & general mapping
# -------------------------
def strip_html_tags(text):
    if not isinstance(text, str):
        return text
    return re.sub(r"<[^>]+>", "", text).strip()

def enrich_record_from_shopify(raw_item: dict, flat_record: dict, default_currency: str = "USD"):
    """
    raw_item: original JSON object for a single product (Shopify-like)
    flat_record: current flattened dict for that product (may be modified in-place or copied)
    This function returns a new dict with enriched fields per ChatGPT spec where possible.
    """
    enriched = dict(flat_record)  # shallow copy

    # Title
    if not enriched.get("title") and raw_item.get("title"):
        enriched["title"] = raw_item.get("title")

    # Description: Shopify uses body_html
    if not enriched.get("description"):
        if raw_item.get("body_html"):
            enriched["description"] = strip_html_tags(raw_item.get("body_html"))

    # Brand: vendor
    if not enriched.get("brand") and raw_item.get("vendor"):
        enriched["brand"] = raw_item.get("vendor")

    # Product category: product_type
    if not enriched.get("product_category") and raw_item.get("product_type"):
        enriched["product_category"] = raw_item.get("product_type")

    # Images: images is typically a list of dicts with src
    images = []
    if "images" in raw_item and isinstance(raw_item["images"], list):
        for img in raw_item["images"]:
            urls = extract_urls_from_any(img)
            for u in urls:
                if u not in images:
                    images.append(u)
    # also check media or gallery keys
    for alt_key in ["media", "photos", "gallery", "assets"]:
        if alt_key in raw_item and isinstance(raw_item[alt_key], list):
            for val in raw_item[alt_key]:
                for u in extract_urls_from_any(val):
                    if u not in images:
                        images.append(u)

    # If flat_record contains arrays of URLs in other keys, include them too
    for k, v in flat_record.items():
        # v might be string like "url1|url2" or list-like; try extract
        if isinstance(v, str) and url_re_global.search(v):
            # split or single
            for u in url_re_global.findall(v):
                if u not in images:
                    images.append(u)

    if images:
        # set image_link to first, additional_image_link to rest joined by comma
        enriched["image_link"] = images[0]
        if len(images) > 1:
            enriched["additional_image_link"] = ", ".join(images[1:])
        else:
            enriched["additional_image_link"] = ""

    # Variants: price, compare_at_price, sku, available, inventory_quantity
    variants = raw_item.get("variants") or []
    # If variants is a list of dicts
    if variants and isinstance(variants, list):
        # pick first variant for canonical fields if not present
        first = variants[0] if variants else {}
        # price
        if not enriched.get("price"):
            price_val = first.get("price") or first.get("presentment_price") or None
            if price_val is not None:
                # price_val might be "99.00" or "99.00 USD" or numeric
                pstr = str(price_val).strip()
                # Does it already have three-letter currency? crude check
                if re.search(r"\b[A-Z]{3}\b", pstr):
                    enriched["price"] = pstr
                else:
                    # attach default currency
                    enriched["price"] = f"{pstr} {default_currency}"
        # sale_price from compare_at_price or compare_at_price > price logic
        if not enriched.get("sale_price"):
            cp = first.get("compare_at_price") or first.get("compare_price")
            if cp:
                pstr = str(cp).strip()
                if re.search(r"\b[A-Z]{3}\b", pstr):
                    enriched["sale_price"] = pstr
                else:
                    enriched["sale_price"] = f"{pstr} {default_currency}"
        # mpn from sku
        if not enriched.get("mpn"):
            sku = first.get("sku") or first.get("mpn")
            if sku:
                enriched["mpn"] = sku
        # availability: if any variant available True => in_stock
        any_available = False
        inv_total = 0
        inv_found = False
        for v in variants:
            if v.get("available") in (True, "true", "True", 1, "1"):
                any_available = True
            # inventory_quantity might be int or string
            iq = v.get("inventory_quantity")
            if iq is not None:
                try:
                    inv_total += int(iq)
                    inv_found = True
                except:
                    pass
        if not enriched.get("availability"):
            enriched["availability"] = "in_stock" if any_available else "out_of_stock"
        if inv_found and not enriched.get("inventory_quantity"):
            enriched["inventory_quantity"] = inv_total

    # product URL: canonical product url may be in "handle" or "url" or "admin_graphql_api_id"
    if not enriched.get("link"):
        if raw_item.get("handle"):
            # no host available; leave as path unless base provided — skip
            # prefer "url" keys if present
            pass
        # try "url" or "product_url"
        for key in ["url","product_url","public_url","permalink"]:
            if raw_item.get(key):
                enriched["link"] = raw_item.get(key)
                break

    # other small mappings
    if not enriched.get("id") and raw_item.get("id"):
        enriched["id"] = raw_item.get("id")

    # tags could form category hints (not applied directly)
    # return enriched dict
    return enriched

# -------------------------
# Validation engine (per-product and feed-level)
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
    spec_attrs = spec_df['Attribute'].tolist()
    spec_norm = [a.lower() for a in spec_attrs]

    for k in record.keys():
        if k.lower() not in spec_norm:
            issues["extras"].append(k)

    for _, srow in spec_df.iterrows():
        attr = srow['Attribute']
        a_norm = attr.lower()
        requirement = parse_requirement(srow.get('Requirement', ""))
        dtype = srow.get('Data Type', "")
        supported = srow.get('Supported Values', "")
        vrules = srow.get('Validation Rules', "")
        dependencies = srow.get('Dependencies', "")

        if a_norm not in record_keys:
            if requirement == "Required":
                issues["missing"].append(attr)
            elif requirement == "Recommended":
                issues["missing"].append(attr + " (recommended)")
            continue

        orig_key = record_keys[a_norm]
        val = record.get(orig_key)

        if val is None or (isinstance(val, str) and str(val).strip() == ""):
            if requirement == "Required":
                issues["empty"].append(attr)
            continue

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

def coverage_summary(feed_df: pd.DataFrame, spec_df: pd.DataFrame):
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

def validate_feed_attribute_level(spec_df: pd.DataFrame, feed_df: pd.DataFrame):
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

def validate_all_products(feed_df: pd.DataFrame, spec_df: pd.DataFrame):
    rows = []
    for idx, item in feed_df.iterrows():
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

# -------------------------
# Row-level failure extraction & mapping preview (used in UI)
# -------------------------
def extract_row_level_failures(feed_df: pd.DataFrame, spec_df: pd.DataFrame):
    rows = []
    feed_cols_norm = [c.lower() for c in feed_df.columns]
    for idx, item in feed_df.iterrows():
        flattened_record = {k: item[k] for k in feed_df.columns}
        prod_id = flattened_record.get("id") or ""
        for _, s in spec_df.iterrows():
            attr = s["Attribute"]
            a_norm = attr.lower()
            requirement = parse_requirement(s.get("Requirement", ""))
            dtype = s.get("Data Type", "")
            supported = s.get("Supported Values", "")
            vrules = s.get("Validation Rules", "")
            if a_norm not in feed_cols_norm:
                if requirement == "Required":
                    rows.append({
                        "record_index": int(idx),
                        "id": prod_id,
                        "attribute": attr,
                        "issue_type": "missing",
                        "details": "Required attribute is missing",
                        "value": ""
                    })
                continue
            orig_col = feed_df.columns[feed_cols_norm.index(a_norm)]
            val = flattened_record.get(orig_col)
            if val is None or (isinstance(val, str) and not str(val).strip()):
                if requirement == "Required":
                    rows.append({
                        "record_index": int(idx),
                        "id": prod_id,
                        "attribute": attr,
                        "issue_type": "empty",
                        "details": "Required value empty",
                        "value": ""
                    })
                continue
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
            ok_s, det_s = check_supported_values(pd.Series([val]), supported)
            if not ok_s:
                rows.append({
                    "record_index": int(idx),
                    "id": prod_id,
                    "attribute": attr,
                    "issue_type": "value",
                    "details": det_s,
                    "value": str(val)
                })
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

def preview_fuzzy_mapping(feed_df: pd.DataFrame, spec_df: pd.DataFrame):
    spec_cols = spec_df['Attribute'].tolist()
    mapping = []
    for c in list(feed_df.columns):
        match = fuzzy_match_column(c, spec_cols)
        mapped = match if match else "(no match)"
        mapping.append({"original": c, "mapped_to": mapped})
    return pd.DataFrame(mapping)

# -------------------------
# -------------------------
# STREAMLIT UI + Main Flow
# -------------------------
st.title("🔎 Auto Schema Feed Validator — Auto-Enriched (Shopify-style support)")
st.write("""
Upload a JSON or XML feed. The tool will:
- auto-detect and flatten records,
- apply fuzzy header mapping,
- auto-enrich Shopify-style fields into ChatGPT spec fields,
- run attribute-level and product-level validation,
- provide per-product expandable UI and downloadable reports.
""")

st.sidebar.header("Settings")
max_expand = st.sidebar.number_input("Max Products to Expand in UI", min_value=10, max_value=5000, value=250, step=10)
allow_all = st.sidebar.checkbox("Allow unlimited expanders (slow for large feeds)", value=False)
default_currency = st.sidebar.text_input("Default currency to append for prices (ISO 4217)", value="USD")

st.sidebar.markdown("---")
st.sidebar.write("Note: Auto-enrichment will try to generate ChatGPT-spec fields from common feed structures (Shopify etc.).")

uploaded = st.file_uploader("📤 Upload Product Feed (JSON or XML)", type=["json","xml"])
if uploaded is None:
    st.info("Upload a feed file to start validation.")
else:
    try:
        uploaded.seek(0)
        ext = uploaded.name.lower().split(".")[-1]
        with st.spinner("Parsing feed..."):
            if ext == "json":
                feed_df, raw_items = json_to_records_with_raw(uploaded)
            else:
                # For XML we don't have raw nested objects; use existing xml->records
                feed_df = xml_to_records(uploaded)
                raw_items = []
        if feed_df is None or feed_df.empty:
            st.error("Parsing failed — no records found. Check feed structure.")
            st.stop()

        st.success(f"Parsed {len(feed_df)} product record(s).")

        # -------------------------
        # Apply fuzzy header mapping first
        # -------------------------
        with st.spinner("Applying fuzzy header mapping…"):
            feed_df = apply_fuzzy_mapping(feed_df, spec_df)

        # Normalize column names (strip)
        feed_df.columns = [str(c).strip() for c in feed_df.columns]

        # -------------------------
        # Enrichment step (Option B)
        # -------------------------
        if raw_items:
            st.info("Auto-enriching records from raw JSON (Shopify-style mapping).")
            enriched_rows = []
            for idx in range(len(feed_df)):
                raw = raw_items[idx] if idx < len(raw_items) else {}
                flat = {k: feed_df.iloc[idx][k] for k in feed_df.columns}
                enriched = enrich_record_from_shopify(raw, flat, default_currency=default_currency)
                enriched_rows.append(enriched)
            # Build DataFrame from enriched_rows (use keys union)
            enriched_df = pd.DataFrame(enriched_rows)
            # Re-apply fuzzy mapping in case enrichment created spec keys differently
            enriched_df = apply_fuzzy_mapping(enriched_df, spec_df)
            feed_df = enriched_df
        else:
            st.info("No raw nested JSON available (XML or single-object JSON). Skipping Shopify auto-enrich.")

        # -------------------------
        # Attribute-level validation
        # -------------------------
        st.markdown("## 1️⃣ Attribute-Level Validation Report")
        with st.spinner("Validating attributes across the full feed…"):
            attr_report = validate_feed_attribute_level(spec_df, feed_df)
        st.dataframe(attr_report, use_container_width=True)
        st.download_button("⬇️ Download Attribute-Level Report (CSV)", attr_report.to_csv(index=False).encode("utf-8"), "attribute_level_report.csv", "text/csv")

        # -------------------------
        # Coverage summary
        # -------------------------
        st.markdown("## 2️⃣ Field Coverage Summary")
        with st.spinner("Computing field coverage…"):
            coverage = coverage_summary(feed_df, spec_df)
        st.dataframe(coverage, use_container_width=True)
        st.download_button("⬇️ Download Field Coverage CSV", coverage.to_csv(index=False).encode("utf-8"), "field_coverage.csv", "text/csv")

        # -------------------------
        # Product-level validation
        # -------------------------
        st.markdown("## 3️⃣ Product-Level Validation (Expandable)")
        with st.spinner("Validating each product…"):
            product_report = validate_all_products(feed_df, spec_df)
        st.success("Product-level validation completed!")
        st.dataframe(product_report.head(200), use_container_width=True)
        st.download_button("⬇️ Download Full Product-Level Issues CSV", product_report.to_csv(index=False).encode("utf-8"), "product_level_issues.csv", "text/csv")

        # Expanders
        st.markdown("## 4️⃣ Inspect Products (Expandable Panels)")
        if allow_all:
            expand_limit = len(feed_df)
        else:
            expand_limit = min(len(feed_df), max_expand)
        st.info(f"Showing first **{expand_limit} products** (use sidebar to increase).")

        for idx in range(expand_limit):
            row = feed_df.iloc[idx]
            flattened_record = {k: row[k] for k in feed_df.columns}
            issues = validate_product_record(flattened_record, spec_df)
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
                st.markdown("**Issues**")
                st.json(issues)
                st.markdown("**Raw flattened/enriched fields for this product**")
                st.dataframe(pd.DataFrame([flattened_record]).T.rename(columns={0: "value"}), use_container_width=True)

        # -------------------------
        # Advanced: row-level failures & mapping preview
        # -------------------------
        st.markdown("---")
        st.markdown("## 5️⃣ Advanced: Row-level failures & mapping preview")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔎 Generate Row-level Failure CSV (detailed)"):
                with st.spinner("Extracting row-level failures..."):
                    failures_df = extract_row_level_failures(feed_df, spec_df)
                if failures_df.empty:
                    st.success("No row-level failures found across scanned attributes.")
                else:
                    st.write(f"Found {len(failures_df)} row-level issues — showing first 200 rows")
                    st.dataframe(failures_df.head(200), use_container_width=True)
                    st.download_button("⬇️ Download row-level failures (CSV)", failures_df.to_csv(index=False).encode("utf-8"), "row_level_failures.csv", "text/csv")
        with col2:
            if st.button("🔁 Preview Fuzzy Column Mapping"):
                with st.spinner("Computing column mapping preview..."):
                    mapping_preview = preview_fuzzy_mapping(feed_df, spec_df)
                st.dataframe(mapping_preview, use_container_width=True)
                st.download_button("⬇️ Download mapping preview (CSV)", mapping_preview.to_csv(index=False).encode("utf-8"), "mapping_preview.csv", "text/csv")

        st.markdown("---")
        st.info("Auto-enrichment completed. Use mapping preview to confirm automatic mappings. If a mapping is wrong, rename fields upstream and re-upload.")
    except Exception as e:
        st.error(f"Failed to parse/validate feed: {e}")
        st.exception(e)

# -------------------------
# requirements (display)
# -------------------------
REQ_TXT = """streamlit
pandas
lxml
"""
st.sidebar.markdown("---")
st.sidebar.subheader("requirements.txt")
st.sidebar.code(REQ_TXT)
