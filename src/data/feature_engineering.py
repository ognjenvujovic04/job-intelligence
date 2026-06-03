import logging
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# Initialize module-level logger
logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool):
    """Internal helper to enable or disable pipeline logging dynamically."""
    if verbose:
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("[%(levelname)s] %(message)s")
            )
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.CRITICAL)


# =========================================================
# 4.1 - SALARY EXTRACTION FROM DESCRIPTION TEXT
# =========================================================

def _clean_and_parse_number(num_str):
    """Remove currency symbols and convert string numbers to floats."""
    if not num_str:
        return np.nan
    clean_str = re.sub(r'[\$,\s]', '', num_str)
    try:
        return float(clean_str)
    except ValueError:
        return np.nan


def _determine_period(match_text):
    """Detect the pay period from a text block context."""
    text = match_text.lower()
    if any(p in text for p in ['hour', 'hr', 'hourly', '/hr']):
        return 'hourly'
    if any(p in text for p in ['day', 'daily', '/day']):
        return 'daily'
    if any(p in text for p in ['year', 'yr', 'annual', 'annually', 'salary', 'compensation', '/yr']):
        return 'yearly'
    return 'unknown'


def _annualize_amount(amount, period):
    """Normalize an amount to a yearly scale based on its pay period."""
    if not amount or period == 'unknown' or not period:
        return np.nan
    if period == 'hourly':
        return amount * 40 * 52   # Standard 2,080-hour work year
    if period == 'daily':
        return amount * 5 * 52    # Standard 260-day work year
    return amount                  # Already yearly


def _extract_salary_from_description(text):
    """
    Extract an annualized salary from free-text job descriptions using
    regex patterns. Returns NaN when no valid salary is found.

    Handles dollar ranges (midpoint), multiple pay periods (hourly, daily,
    yearly), and guards against false positives like 401(k) mentions.
    """
    if not isinstance(text, str) or not text.strip():
        return np.nan

    # Base regex patterns; negative lookahead protects against 401k variations
    money_num = r'\d{1,3}(?:,\d{3})*(?:\.\d{2})?(?!\s*k\b)'
    money_pattern = rf'\$\s*{money_num}(?:\s*(?:-|to)\s*\$?\s*{money_num})?'

    keywords = r'(?:pay(?:| range)?|salary|wage|hourly rate|compensation|starting at|from)'
    timeframes = r'(?:hour|hr|hourly|year|yr|annually|per year|annual salary|one day|day|daily)'

    patterns = [
        rf'{keywords}\s*[:\-]?\s*(?:from\s*)?({money_pattern}(?:\s*(?:/|per\s*)?{timeframes})?)',
        rf'({money_pattern}\s*(?:/|per\s*)?{timeframes})',
        rf'({money_pattern})',
    ]

    master_regex = re.compile('|'.join(patterns), re.IGNORECASE)
    annualized_candidates = []

    for match_obj in master_regex.finditer(text):
        clean_match = next((g.strip() for g in match_obj.groups() if g), "")
        if not clean_match:
            continue

        numbers = re.findall(
            r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)(?!\s*k\b)', clean_match
        )
        parsed_vals = [
            _clean_and_parse_number(n)
            for n in numbers
            if _clean_and_parse_number(n) is not None
        ]

        if parsed_vals:
            start_idx = max(0, match_obj.start() - 15)
            end_idx = min(len(text), match_obj.end() + 15)
            context_window = text[start_idx:end_idx]

            match_period = _determine_period(context_window)
            match_base_value = sum(parsed_vals) / len(parsed_vals)
            match_annualized = round(
                _annualize_amount(match_base_value, match_period), 2
            )

            if match_annualized and not np.isnan(match_annualized):
                annualized_candidates.append(match_annualized)

    if not annualized_candidates:
        return np.nan

    # Return the median candidate to reduce sensitivity to outlier matches
    return float(np.median(annualized_candidates))


def extract_salary_features(df):
    """
    Extract salary values from description text and cap outliers.

    Applies regex-based salary extraction to the 'description' column,
    then caps results at the 99th percentile to remove false positives.

    Parameters:
        df (pd.DataFrame): Feature dataframe with 'description' column.

    Returns:
        pd.DataFrame
    """
    logger.info("Extracting salary values from description text")
    tqdm.pandas(desc="Extracting salaries from descriptions")
    df["extracted_salary"] = df["description"].progress_apply(_extract_salary_from_description)

    valid_extracted = df.loc[df["extracted_salary"] > 0, "extracted_salary"]
    if not valid_extracted.empty:
        cap = np.percentile(valid_extracted, 99)
        df.loc[df["extracted_salary"] > cap, "extracted_salary"] = np.nan

    extracted_count = df["extracted_salary"].notna().sum()
    logger.info(f"  Extracted {extracted_count:,} salary values from descriptions")

    return df


# =========================================================
# 4.2 - DOMAIN EXTRACTION (from pre-computed similarities)
# =========================================================

def merge_domain_features(df, domain_path, similarity_threshold=0.35):
    """
    Load pre-computed domain similarity scores and derive domain labels.

    Reads cosine-similarity scores between job descriptions and 13
    industry-domain prototypes (produced externally by a BERT model).
    Assigns the highest-scoring domain as a label, or 'Unknown' when
    the best score falls below the threshold. One-hot encodes the
    resulting labels and keeps the best similarity as a continuous feature.

    Parameters:
        df (pd.DataFrame): Feature dataframe with 'job_id' column.
        domain_path (str): Path to domain_probabilities.csv.
        similarity_threshold (float): Minimum similarity to assign a domain.

    Returns:
        pd.DataFrame
    """
    if not os.path.exists(domain_path):
        logger.warning(f"Domain file not found at '{domain_path}', skipping domain features")
        return df

    logger.info(f"Loading domain similarities from '{domain_path}'")
    df_domains = pd.read_csv(domain_path)

    sim_cols = [c for c in df_domains.columns if c.startswith("domain_sim_")]
    domain_names = [c.replace("domain_sim_", "") for c in sim_cols]

    sim_values = df_domains[sim_cols].values
    best_idx = sim_values.argmax(axis=1)
    best_score = sim_values[np.arange(len(sim_values)), best_idx]

    df_domains["domain"] = np.where(
        best_score >= similarity_threshold,
        np.array(domain_names)[best_idx],
        "Unknown",
    )
    df_domains["domain_similarity"] = best_score

    df = df.merge(
        df_domains[["job_id", "domain", "domain_similarity"]],
        on="job_id",
        how="left",
    )

    domain_dummies = pd.get_dummies(df["domain"], prefix="domain")
    df = pd.concat([df, domain_dummies], axis=1)

    logger.info(
        f"  Domain columns added | unique domains: {df['domain'].nunique()} | "
        f"shape: {df.shape}"
    )
    return df


# =========================================================
# 4.3 - TITLE-BASED EXPERIENCE LEVEL EXTRACTION
# =========================================================

TITLE_EXPERIENCE_PATTERNS = [
    (0, re.compile(r'\b(?:intern|internship)\b', re.IGNORECASE)),
    (1, re.compile(r'\b(?:junior|jr\.?|entry[\s-]?level|trainee|apprentice)\b', re.IGNORECASE)),
    (2, re.compile(r'\b(?:associate)\b', re.IGNORECASE)),
    (3, re.compile(r'\b(?:senior|sr\.?|staff|principal)\b', re.IGNORECASE)),
    (4, re.compile(r'\b(?:director)\b', re.IGNORECASE)),
    (5, re.compile(
        r'\b(?:vice[\s-]?president|vp|c[eoftim]o|chief\s\w+\sofficer|president|partner)\b',
        re.IGNORECASE,
    )),
]

FALSE_POSITIVE_GUARDS = {
    5: re.compile(r'\b(?:executive\s+(?:assistant|secretary|coordinator|admin))\b', re.IGNORECASE),
    3: re.compile(r'\b(?:senior\s+(?:living|care|center|community|services))\b', re.IGNORECASE),
    2: re.compile(
        r'\b(?:'
        r'(?:sales|retail|warehouse|fulfillment|store|stock|team|pickup|delivery|deli)\s+associate'
        r'|associate\s+(?:dentist|veterinarian|pastor|broker)'
        r')\b',
        re.IGNORECASE,
    ),
}

EXPERIENCE_LEVEL_NAMES = {
    0: 'Internship',
    1: 'Entry level',
    2: 'Associate',
    3: 'Mid-Senior level',
    4: 'Director',
    5: 'Executive',
}


def _extract_experience_from_title(title):
    """
    Return an ordinal experience level (0-5) from a job title using regex.

    Scans from highest seniority downward so compound titles like
    'Senior Director' resolve to Director (4) rather than Mid-Senior (3).
    Returns None when no keyword matches.
    """
    if not isinstance(title, str) or not title.strip():
        return None

    for level, pattern in reversed(TITLE_EXPERIENCE_PATTERNS):
        if pattern.search(title):
            guard = FALSE_POSITIVE_GUARDS.get(level)
            if guard and guard.search(title):
                continue
            return level

    return None


def fill_experience_from_title(df):
    """
    Fill missing formatted_experience_level values using title keywords.

    Parameters:
        df (pd.DataFrame): Feature dataframe with 'title' and
            'formatted_experience_level' columns.

    Returns:
        pd.DataFrame
    """
    logger.info("Extracting experience level from job titles")

    df['_title_exp_extracted'] = df['title'].apply(_extract_experience_from_title)
    df['_title_exp_label'] = df['_title_exp_extracted'].map(EXPERIENCE_LEVEL_NAMES)

    new_fills = df['formatted_experience_level'].isna() & df['_title_exp_label'].notna()
    fill_count = new_fills.sum()

    df['formatted_experience_level'] = df['formatted_experience_level'].fillna(
        df['_title_exp_label']
    )

    df.drop(columns=['_title_exp_extracted', '_title_exp_label'], inplace=True)

    remaining_nan = df['formatted_experience_level'].isna().sum()
    logger.info(
        f"  Filled {fill_count:,} missing values from title extraction | "
        f"remaining NaN: {remaining_nan:,}"
    )
    return df


# =========================================================
# 4.4 - TEXT LENGTH FEATURES
# =========================================================

def add_text_length_features(df):
    """
    Add title_length and desc_word_count features.

    Parameters:
        df (pd.DataFrame): Feature dataframe with 'title' and 'description'.

    Returns:
        pd.DataFrame
    """
    logger.info("Adding text length features (title_length, desc_word_count)")
    df['title_length'] = df['title'].str.len()
    df['desc_word_count'] = df['description'].fillna('').str.split().str.len()
    return df


# =========================================================
# 5.1 - TEMPORAL FEATURES
# =========================================================

def add_temporal_features(df):
    """
    Derive temporal features from epoch-ms timestamps.

    Produces:
        - posting_duration_days: days between listing and expiry
        - is_long_posting: binary flag for postings with > 60-day duration
        - listed_day_of_week: day of week the posting went live (0=Mon)

    Parameters:
        df (pd.DataFrame): Feature dataframe with 'listed_time' and 'expiry'.

    Returns:
        pd.DataFrame
    """
    logger.info("Adding temporal features (is_long_posting, listed_day_of_week)")

    df['listed_dt'] = pd.to_datetime(df['listed_time'], unit='ms', errors='coerce')
    df['expiry_dt'] = pd.to_datetime(df['expiry'], unit='ms', errors='coerce')

    df['posting_duration_days'] = (
        (df['expiry_dt'] - df['listed_dt']).dt.total_seconds() / 86400
    ).round(2)

    df['is_long_posting'] = (df['posting_duration_days'] > 60).astype(int)
    df['listed_day_of_week'] = df['listed_dt'].dt.dayofweek

    long_n = df['is_long_posting'].sum()
    logger.info(f"  Extended postings (>60 days): {long_n:,} ({long_n / len(df) * 100:.1f}%)")

    return df


# =========================================================
# 5.2 - ENGAGEMENT FEATURES
# =========================================================

def add_engagement_features(df):
    """
    Log-transform views/applies and derive apply_rate.

    Produces:
        - log_views: log(1 + views)
        - log_applies: log(1 + applies)
        - apply_rate: applies / views (NaN where views == 0)

    Parameters:
        df (pd.DataFrame): Feature dataframe with 'views' and 'applies'.

    Returns:
        pd.DataFrame
    """
    logger.info("Adding engagement features (log_views, log_applies, apply_rate)")

    df['log_views'] = np.log1p(df['views'])
    df['log_applies'] = np.log1p(df['applies'])
    df['apply_rate'] = df['applies'] / df['views'].replace(0, np.nan)

    return df


# =========================================================
# 5.3 - SALARY CONSOLIDATION
# =========================================================

def consolidate_salary(df, floor=10_000, cap_percentile=0.96):
    """
    Clean extracted salaries and fill missing normalized_salary values.

    Filters extracted salaries by a floor and a percentile cap, then
    merges them into normalized_salary where the original is missing.

    Parameters:
        df (pd.DataFrame): Feature dataframe with 'normalized_salary'
            and 'extracted_salary'.
        floor (float): Minimum salary to keep (removes false positives).
        cap_percentile (float): Upper percentile cap for extracted values.

    Returns:
        pd.DataFrame
    """
    logger.info("Consolidating salary: merging extracted values into normalized_salary")

    raw_extracted = df['extracted_salary'].dropna()
    cap = raw_extracted.quantile(cap_percentile)

    df['extracted_salary_clean'] = df['extracted_salary'].copy()
    df.loc[df['extracted_salary_clean'] < floor, 'extracted_salary_clean'] = np.nan
    df.loc[df['extracted_salary_clean'] > cap, 'extracted_salary_clean'] = np.nan

    before_missing = df['normalized_salary'].isna().sum()

    df['salary_source'] = np.where(
        df['normalized_salary'].notna(), 'original',
        np.where(df['extracted_salary_clean'].notna(), 'extracted', 'missing'),
    )
    df['normalized_salary'] = df['normalized_salary'].fillna(df['extracted_salary_clean'])

    after_missing = df['normalized_salary'].isna().sum()
    filled = before_missing - after_missing

    logger.info(
        f"  Filled {filled:,} salaries from text | "
        f"missing before: {before_missing:,} | after: {after_missing:,}"
    )
    return df


# =========================================================
# 6.1 - ONE-HOT ENCODING (low cardinality)
# =========================================================

def one_hot_encode(df, columns=None):
    """
    One-hot encode low-cardinality categorical columns.

    Drops singleton dummy columns (single occurrence in dataset).

    Parameters:
        df (pd.DataFrame): Feature dataframe.
        columns (list): Columns to one-hot encode.

    Returns:
        pd.DataFrame
    """
    default_columns = ['pay_period', 'formatted_work_type', 'application_type']
    columns = columns if columns is not None else default_columns
    existing = [c for c in columns if c in df.columns]

    logger.info(f"One-hot encoding columns: {existing}")

    for col in existing:
        dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
        df = pd.concat([df, dummies], axis=1)
        df.drop(columns=[col], inplace=True)

    # Drop singleton OHE columns (only 1 occurrence)
    if 'application_type_UnknownApply' in df.columns:
        if df['application_type_UnknownApply'].sum() <= 1:
            df.drop(columns=['application_type_UnknownApply'], inplace=True)
            logger.info("  Dropped 'application_type_UnknownApply' (singleton)")

    return df


# =========================================================
# 6.2 - ORDINAL ENCODING: experience_level
# =========================================================

EXPERIENCE_ORDER = {
    'Internship':       0,
    'Entry level':      1,
    'Associate':        2,
    'Mid-Senior level': 3,
    'Director':         4,
    'Executive':        5,
}


def ordinal_encode_experience(df):
    """
    Map formatted_experience_level to ordinal integers (0-5).

    Parameters:
        df (pd.DataFrame): Feature dataframe with 'formatted_experience_level'.

    Returns:
        pd.DataFrame
    """
    logger.info("Ordinal encoding experience level (0=Internship .. 5=Executive)")

    df['experience_level_ord'] = df['formatted_experience_level'].map(EXPERIENCE_ORDER)

    missing = df['experience_level_ord'].isna().sum()
    logger.info(f"  NaN after mapping: {missing:,}")

    return df


# =========================================================
# 6.3 - BINARY ENCODING: remote_allowed
# =========================================================

def encode_remote_allowed(df):
    """
    Fill NaN in remote_allowed with 0 and cast to int.

    All observed values are 1 (True); ~87% are missing and treated as
    0 (not remote).

    Parameters:
        df (pd.DataFrame): Feature dataframe with 'remote_allowed'.

    Returns:
        pd.DataFrame
    """
    if 'remote_allowed' in df.columns:
        logger.info("Binary encoding remote_allowed (NaN -> 0)")
        df['remote_allowed'] = df['remote_allowed'].fillna(0).astype(int)
    return df


# =========================================================
# 7.1 - DROP REDUNDANT COLUMNS
# =========================================================

def drop_redundant_columns(df):
    """
    Drop columns that have been consumed by derived features.

    Removes identifiers, raw text, intermediate columns, superseded
    originals, raw timestamps, raw salary components, and raw engagement
    counts. Keeps high-cardinality columns for later encoding.

    Parameters:
        df (pd.DataFrame): Feature dataframe.

    Returns:
        pd.DataFrame
    """
    cols_to_drop = [
        # identifiers
        'job_id',
        # raw text (features already extracted)
        'title', 'description',
        # intermediate / temp columns
        'extracted_salary', 'extracted_salary_clean', 'salary_source',
        'listed_dt', 'expiry_dt', 'posting_duration_days',
        # superseded by ordinal encoding
        'formatted_experience_level',
        # raw timestamps (temporal features derived)
        'original_listed_time', 'listed_time', 'expiry',
        # raw salary components (consolidated into normalized_salary)
        'min_salary', 'max_salary', 'med_salary', 'currency',
        # raw engagement (log versions kept)
        'views', 'applies',
        # domain label (one-hot encoded already)
        'domain',
    ]

    existing_drops = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=existing_drops, inplace=True)

    logger.info(f"Dropped {len(existing_drops)} redundant columns | shape: {df.shape}")
    return df


# =========================================================
# 7.2 - TRAIN / TEST SPLIT
# =========================================================

def split_train_test(df, test_size=0.20, random_state=42):
    """
    Split the feature dataframe into train and test sets.

    Parameters:
        df (pd.DataFrame): Feature dataframe.
        test_size (float): Fraction of data for the test set.
        random_state (int): Random seed for reproducibility.

    Returns:
        (pd.DataFrame, pd.DataFrame): train and test dataframes.
    """
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
    )

    df_train = df.loc[train_idx].copy()
    df_test = df.loc[test_idx].copy()

    logger.info(
        f"Train/test split ({1 - test_size:.0%}/{test_size:.0%}) | "
        f"train: {len(df_train):,} | test: {len(df_test):,}"
    )
    return df_train, df_test


# =========================================================
# 8.1 - STATE EXTRACTION + TARGET ENCODING
# =========================================================

REGIONAL_PATTERNS = {
    'ny': ['new york', 'buffalo-niagara falls area', 'utica-rome area'],
    'ca': ['san francisco', 'bay area', 'los angeles', 'sacramento',
           'san diego', 'silicon valley'],
    'tx': ['dallas', 'fort worth', 'houston', 'metroplex', 'texas', 'austin',
           'greater corpus christi area', 'san angelo area'],
    'dc': ['washington dc', 'baltimore'],
    'ga': ['atlanta', 'gainesville metropolitan area', 'greater macon'],
    'tn': ['nashville', 'memphis', 'knoxville metropolitan area'],
    'co': ['denver'],
    'ma': ['boston', 'greater boston',
           'springfield, massachusetts metropolitan area'],
    'fl': ['miami', 'fort lauderdale', 'orlando', 'metro jacksonville'],
    'az': ['phoenix'],
    'il': ['chicago', 'illinois', 'peoria metropolitan area'],
    'mn': ['minneapolis', 'st. paul', 'minnesota'],
    'mi': ['detroit', 'grand rapids', 'michigan'],
    'or': ['oregon', 'portland'],
    'ut': ['salt lake city', 'utah'],
    'wa': ['seattle', 'washington state'],
    'mo': ['kansas city', 'st. louis', 'missouri'],
    'ks': ['kansas metropolitan', 'topeka metropolitan area'],
    'in': ['indianapolis', 'indiana'],
    'nc': ['charlotte', 'greensboro', 'winston-salem', 'high point',
           'north carolina', 'raleigh-durham-chapel hill area'],
    'oh': ['cincinnati', 'cleveland', 'ohio'],
    'pa': ['pittsburgh', 'philadelphia', 'pennsylvania'],
    'ia': ['des moines', 'iowa'],
    'va': ['richmond', 'virginia'],
    'ar': ['little rock', 'arkansas'],
    'la': ['baton rouge', 'new orleans', 'louisiana'],
    'ct': ['hartford', 'connecticut'],
    'wi': ['appleton', 'oshkosh', 'neenah', 'eau claire', 'menomonie',
           'wisconsin', 'greater milwaukee', 'greater madison area'],
    'ne': ['omaha metropolitan area', 'lincoln, nebraska metropolitan area'],
    'nv': ['las vegas metropolitan area', 'greater reno area'],
    'sc': ['columbia, south carolina metropolitan area',
           'charleston, south carolina metropolitan area',
           'greenville-spartanburg-anderson, south carolina area'],
    'al': ['greater birmingham, alabama area',
           'huntsville-decatur-albertville area'],
    'ky': ['louisville metropolitan area'],
    'ok': ['oklahoma city metropolitan area'],
    'de': ['greater wilmington area'],
    'ak': ['greater anchorage area'],
    'id': ['boise metropolitan area', 'greater idaho falls'],
    'hi': ['maui', 'honolulu metropolitan area'],
    'nm': ['albuquerque-santa fe metropolitan area'],
}


def _extract_state(loc):
    """
    Extract a US state abbreviation from a location string.

    Uses a three-tier approach:
        1. Comma-split: 'City, ST' format
        2. Regional keyword matching against REGIONAL_PATTERNS
        3. Broad US labels ('united states', 'namer') map to 'other';
           unresolvable entries map to 'noise'
    """
    if pd.isna(loc):
        return np.nan

    loc_clean = str(loc).strip().lower()

    # Tier 1: "City, ST" format
    if ',' in loc_clean:
        last_part = loc_clean.split(',')[-1].strip()
        if len(last_part) == 2 and last_part.isalpha():
            return last_part

    # Tier 2: regional keyword lookup
    for state_code, keywords in REGIONAL_PATTERNS.items():
        for kw in keywords:
            if kw in loc_clean:
                return state_code

    # Tier 3: broad US labels
    if any(label in loc_clean for label in ['united states', 'namer', 'nationwide']):
        return 'other'

    return 'noise'


def encode_location(df_train, df_test, smoothing=20):
    """
    Extract state from location and apply target encoding against salary.

    Noise rows (international/unresolvable) are dropped from train and
    remapped to NaN in test (receiving the global salary mean).

    Parameters:
        df_train (pd.DataFrame): Training set.
        df_test (pd.DataFrame): Test set.
        smoothing (int): Smoothing factor for target encoding.

    Returns:
        (pd.DataFrame, pd.DataFrame)
    """
    logger.info("Encoding location: state extraction + salary target encoding")

    df_train['state'] = df_train['location'].apply(_extract_state)
    df_test['state'] = df_test['location'].apply(_extract_state)

    # Handle noise rows
    noise_train = (df_train['state'] == 'noise').sum()
    noise_test = (df_test['state'] == 'noise').sum()

    df_train = df_train[df_train['state'] != 'noise'].copy()
    df_test.loc[df_test['state'] == 'noise', 'state'] = np.nan

    logger.info(
        f"  Noise rows: train dropped {noise_train}, test remapped {noise_test} to NaN"
    )

    # Target encoding (fit on train only)
    global_mean = df_train['normalized_salary'].mean()

    state_stats = (
        df_train
        .groupby('state')['normalized_salary']
        .agg(['mean', 'count'])
    )
    state_stats['encoded'] = (
        (state_stats['count'] * state_stats['mean'] + smoothing * global_mean)
        / (state_stats['count'] + smoothing)
    )
    state_encoding = state_stats['encoded']

    df_train['state_salary_enc'] = df_train['state'].map(state_encoding).fillna(global_mean)
    df_test['state_salary_enc'] = df_test['state'].map(state_encoding).fillna(global_mean)

    logger.info(f"  States encoded: {len(state_encoding)} | global mean: ${global_mean:,.0f}")

    df_train.drop(columns=['state', 'location'], inplace=True)
    df_test.drop(columns=['state', 'location'], inplace=True)

    return df_train, df_test


# =========================================================
# 8.2 - FIPS TARGET ENCODING
# =========================================================

def encode_fips(df_train, df_test, smoothing=20):
    """
    Target-encode FIPS codes against normalized_salary.

    Fit on train only. Unseen/missing FIPS in test receive the global
    salary mean. Also drops zip_code (overlaps with fips).

    Parameters:
        df_train (pd.DataFrame): Training set.
        df_test (pd.DataFrame): Test set.
        smoothing (int): Smoothing factor.

    Returns:
        (pd.DataFrame, pd.DataFrame)
    """
    logger.info("Target encoding FIPS codes against salary")

    global_mean = df_train['normalized_salary'].mean()

    fips_stats = (
        df_train
        .groupby('fips')['normalized_salary']
        .agg(['mean', 'count'])
    )
    fips_stats['encoded'] = (
        (fips_stats['count'] * fips_stats['mean'] + smoothing * global_mean)
        / (fips_stats['count'] + smoothing)
    )
    fips_encoding = fips_stats['encoded']

    df_train['fips_salary_enc'] = df_train['fips'].map(fips_encoding).fillna(global_mean)
    df_test['fips_salary_enc'] = df_test['fips'].map(fips_encoding).fillna(global_mean)

    logger.info(f"  Unique FIPS in train: {len(fips_encoding):,}")

    for col in ['fips', 'zip_code']:
        if col in df_train.columns:
            df_train.drop(columns=[col], inplace=True)
        if col in df_test.columns:
            df_test.drop(columns=[col], inplace=True)

    return df_train, df_test


# =========================================================
# 8.3 - FREQUENCY ENCODING (company + posting domain)
# =========================================================

def frequency_encode(df_train, df_test, configs=None):
    """
    Frequency-encode high-cardinality columns using train-set counts.

    Unseen categories in test and NaN values both receive 1 (conservative
    minimum frequency).

    Parameters:
        df_train (pd.DataFrame): Training set.
        df_test (pd.DataFrame): Test set.
        configs (dict): Mapping of {raw_column: new_column_name}.

    Returns:
        (pd.DataFrame, pd.DataFrame)
    """
    default_configs = {
        'company_id':     'company_freq',
        'posting_domain': 'posting_domain_freq',
    }
    configs = configs if configs is not None else default_configs

    logger.info(f"Frequency encoding: {list(configs.keys())}")

    for raw_col, new_col in configs.items():
        if raw_col not in df_train.columns:
            continue

        freq_map = df_train[raw_col].value_counts()

        df_train[new_col] = df_train[raw_col].map(freq_map).fillna(1)
        df_test[new_col] = df_test[raw_col].map(freq_map)

        # Unseen categories in test get 1
        unseen_mask = df_test[new_col].isna() & df_test[raw_col].notna()
        df_test.loc[unseen_mask, new_col] = 1
        df_test[new_col] = df_test[new_col].fillna(1)

        logger.info(f"  {raw_col} -> {new_col} | unique in train: {len(freq_map):,}")

    # Drop raw columns
    drop_cols = list(configs.keys()) + ['company_name']
    for col in drop_cols:
        if col in df_train.columns:
            df_train.drop(columns=[col], inplace=True)
        if col in df_test.columns:
            df_test.drop(columns=[col], inplace=True)

    return df_train, df_test


# =========================================================
# 9 - EXPORT
# =========================================================

def save_feature_matrices(
    df_train,
    df_test,
    train_path="../../data/processed/feature_matrix_train.csv",
    test_path="../../data/processed/feature_matrix_test.csv",
):
    """
    Save train and test feature matrices to CSV.

    Parameters:
        df_train (pd.DataFrame): Training feature matrix.
        df_test (pd.DataFrame): Test feature matrix.
        train_path (str): Output path for training set.
        test_path (str): Output path for test set.
    """
    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    

    logger.info(
        f"Exported train: {train_path} ({df_train.shape[0]:,} x {df_train.shape[1]})"
    )
    logger.info(
        f"Exported test:  {test_path} ({df_test.shape[0]:,} x {df_test.shape[1]})"
    )


# =========================================================
# COMPLETE PIPELINE
# =========================================================

def feature_engineering_pipeline(
    input_path="../../data/processed/cleaned_job_postings.csv",
    domain_path="../../data/precomputed/domain_probabilities.csv",
    train_output="../../data/processed/feature_matrix_train.csv",
    test_output="../../data/processed/feature_matrix_test.csv",
    test_size=0.20,
    random_state=42,
    smoothing=20,
    verbose=True,
):
    """
    Complete feature engineering pipeline.

    Loads the cleaned dataset, applies all feature transformations
    (text extraction, encoding, temporal and engagement features),
    splits into train/test, encodes high-cardinality features without
    leakage, and exports the final feature matrices.

    Parameters:
        input_path (str): Path to cleaned CSV from preprocessing.
        domain_path (str): Path to domain_probabilities.csv.
        train_output (str): Output path for training feature matrix.
        test_output (str): Output path for test feature matrix.
        test_size (float): Test set fraction.
        random_state (int): Random seed.
        smoothing (int): Smoothing factor for target encoding.
        verbose (bool): If True, prints step-by-step progress.

    Returns:
        (pd.DataFrame, pd.DataFrame): train and test feature matrices.
    """
    _configure_logging(verbose)

    df = pd.read_csv(input_path)
    logger.info(f"Pipeline started | input shape: {df.shape}")

    # -- Text feature engineering (Sections 4.1 - 4.4) --
    df_fe = df.copy()
    df_fe = extract_salary_features(df_fe)
    df_fe = merge_domain_features(df_fe, domain_path)
    df_fe = fill_experience_from_title(df_fe)
    df_fe = add_text_length_features(df_fe)

    # -- Numerical features (Sections 5.1 - 5.3) --
    df_fe = add_temporal_features(df_fe)
    df_fe = add_engagement_features(df_fe)
    df_fe = consolidate_salary(df_fe)

    # -- Categorical encoding (Sections 6.1 - 6.3) --
    df_fe = one_hot_encode(df_fe)
    df_fe = ordinal_encode_experience(df_fe)
    df_fe = encode_remote_allowed(df_fe)

    # -- Drop redundant columns and split (Section 7) --
    df_fe = drop_redundant_columns(df_fe)
    df_train, df_test = split_train_test(df_fe, test_size, random_state)

    # -- High-cardinality encoding (Section 8, fit on train only) --
    df_train, df_test = encode_location(df_train, df_test, smoothing)
    df_train, df_test = encode_fips(df_train, df_test, smoothing)
    df_train, df_test = frequency_encode(df_train, df_test)

    # -- Export (Section 9) --
    save_feature_matrices(df_train, df_test, train_output, test_output)

    logger.info(
        f"Pipeline finished | train: {df_train.shape} | test: {df_test.shape}"
    )
    return df_train, df_test


# =========================================================
# RUN SCRIPT
# =========================================================

if __name__ == "__main__":
    feature_engineering_pipeline()