"""
Client-side smoke script for the inference API (NOT a pytest test).

Start the server yourself first, e.g.:
    python -m src.api

Then call one, several, or all endpoints against the hardcoded sample postings:
    python scripts/api_client.py                       # all endpoints
    python scripts/api_client.py --task salary         # just salary
    python scripts/api_client.py --task experience salary
    python scripts/api_client.py --task all --no-summarize   # all except T5
    python scripts/api_client.py --task salary --time         # report request latency
    python scripts/api_client.py --base-url http://localhost:8001

Edit SAMPLE_POSTINGS below to test your own records. Uses only the standard
library so there is nothing extra to install.
"""

import argparse
import json
import time
import urllib.error
import urllib.request

# --- Hardcoded request body: edit these to test individual records ----------
# Each record mirrors the raw data/raw/postings.csv schema (all 31 columns).
# Only job_id is required by the API; every other field is optional and imputed
# downstream, but a full record exercises the pipeline the way training data did.
# Timestamps are epoch milliseconds (as in the raw dataset).
SAMPLE_POSTINGS = [
    {
        "job_id": 1,
        "company_name": "Acme Analytics",
        "title": "Senior Data Scientist",
         "description": (
            "We are hiring a senior data scientist to design, train, and deploy "
            "machine learning models for our analytics platform. You will work "
            "with Python, SQL, and cloud infrastructure (AWS) and partner with "
            "engineering to ship models to production."
        ),
        "max_salary": 160000,
        "pay_period": "YEARLY",
        "location": "San Francisco, CA",
        "company_id": 12345,
        "views": 120,
        "med_salary": None,
        "min_salary": 120000,
        "formatted_work_type": "Full-time",
        "applies": 150,
        "original_listed_time": 1700000000000,
        "remote_allowed": 1,
        "job_posting_url": "https://example.com/job/1",
        "application_url": "https://example.com/apply/1",
        "application_type": "OffsiteApply",
        "expiry": 1702592000000,
        "closed_time": None,
        "formatted_experience_level": "Mid-Senior level",
        "skills_desc": None,
        "listed_time": 1700000000000,
        "posting_domain": "acme.com",
        "sponsored": 0,
        "work_type": "FULL_TIME",
        "currency": "USD",
        "compensation_type": "BASE_SALARY",
        "normalized_salary": 140000,
        "zip_code": 94105,
        "fips": 6075,
    },
    # {
    #     "job_id": 2,
    #     "company_name": "City General Hospital",
    #     "title": "Registered Nurse",
    #     "description": (
    #         "company: city general hospital\n\n"
    #         "department: medical-surgical unit\n\n"
    #         "location: chicago, il\n\n"
    #         "shift: full-time, 12-hour day and night rotations, including weekends "
    #         "and holidays\n\n"
    #         "compensation: $75,000 - $98,000 per year, depending on experience\n\n"
    #         "about the role:\n\n"
    #         "city general hospital is seeking a compassionate and dependable "
    #         "registered nurse (rn) to join our busy medical-surgical unit. as a "
    #         "front-line caregiver, you will provide direct, hands-on patient care, "
    #         "coordinate closely with physicians and the interdisciplinary care team, "
    #         "and serve as an advocate for patients and their families throughout their "
    #         "stay. we are looking for nurses who combine strong clinical judgment with "
    #         "genuine empathy and a commitment to safe, high-quality care.\n\n"
    #         "key responsibilities:\n\n"
    #         "assess, plan, implement, and evaluate individualized nursing care plans "
    #         "for an assigned group of patients. administer medications and treatments "
    #         "accurately and safely, in accordance with physician orders and hospital "
    #         "policy. monitor and document patient vital signs, symptoms, and responses "
    #         "to treatment in the electronic health record. collaborate with physicians, "
    #         "case managers, therapists, and pharmacists to coordinate care and discharge "
    #         "planning. educate patients and families on diagnoses, medications, and "
    #         "post-discharge self-care. respond promptly to changes in patient condition "
    #         "and escalate concerns appropriately. maintain a clean, safe, and organized "
    #         "care environment and adhere to all infection-control and patient-safety "
    #         "protocols.\n\n"
    #         "qualifications:\n\n"
    #         "current, active registered nurse (rn) license in the state of illinois "
    #         "(or eligibility for licensure by endorsement). associate or bachelor of "
    #         "science in nursing (adn or bsn) from an accredited program; bsn preferred. "
    #         "current bls certification; acls preferred or required within 90 days of "
    #         "hire. minimum of one year of acute-care or med-surg nursing experience "
    #         "preferred; new graduates with strong clinical rotations will be considered. "
    #         "strong assessment, critical-thinking, and communication skills. ability to "
    #         "work effectively in a fast-paced, high-acuity environment and to remain "
    #         "calm under pressure.\n\n"
    #         "benefits:\n\n"
    #         "comprehensive medical, dental, and vision insurance, a retirement savings "
    #         "plan with employer contribution, generous paid time off, shift "
    #         "differentials for nights and weekends, tuition reimbursement, and ongoing "
    #         "professional-development support. city general hospital is an equal "
    #         "opportunity employer committed to building an inclusive and diverse "
    #         "workforce; candidates of all backgrounds are encouraged to apply."
    #     ),
    #     # Deliberately anomalous salary to exercise /detect/anomalies.
    #     "max_salary": 1000000000,
    #     "pay_period": "YEARLY",
    #     "location": "Chicago, IL",
    #     "company_id": 67890,
    #     "views": 340,
    #     "med_salary": None,
    #     "min_salary": 75000,
    #     "formatted_work_type": "Full-time",
    #     "applies": 520,
    #     "original_listed_time": 1700100000000,
    #     "remote_allowed": 0,
    #     "job_posting_url": "https://example.com/job/2",
    #     "application_url": "https://example.com/apply/2",
    #     "application_type": "SimpleOnsiteApply",
    #     "expiry": 1702692000000,
    #     "closed_time": None,
    #     "formatted_experience_level": "Entry level",
    #     "skills_desc": None,
    #     "listed_time": 1700100000000,
    #     "posting_domain": "citygeneralhospital.org",
    #     "sponsored": 0,
    #     "work_type": "FULL_TIME",
    #     "currency": "USD",
    #     "compensation_type": "BASE_SALARY",
    #     "normalized_salary": 950000000,
    #     "zip_code": 60601,
    #     "fips": 17031,
    # },
]

# Task name -> route path.
ROUTES = {
    "experience": "/predict/experience-level",
    "salary": "/predict/salary",
    "clusters": "/predict/clusters",
    "anomalies": "/detect/anomalies",
    "summarize": "/summarize",
}


def call(base_url, path, body):
    """POST `body` as JSON to base_url+path.

    Return (status, parsed_json|text, elapsed_seconds), where elapsed measures
    the full round trip (send + server processing + response read).
    """
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            return resp.status, payload, time.perf_counter() - start
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8"), time.perf_counter() - start
    except urllib.error.URLError as e:
        reason = f"connection failed: {e.reason} (is the server running?)"
        return None, reason, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        nargs="+",
        choices=list(ROUTES) + ["all"],
        default=["all"],
        help="which endpoint(s) to call (default: all)",
    )
    parser.add_argument(
        "--no-summarize",
        action="store_true",
        help="skip /summarize even when 'all' is selected (avoids the slow T5 load)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="server base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--time",
        action="store_true",
        help="report how long each request took (full round trip, in seconds)",
    )
    args = parser.parse_args()

    tasks = list(ROUTES) if "all" in args.task else list(dict.fromkeys(args.task))
    if args.no_summarize and "summarize" in tasks:
        tasks.remove("summarize")

    body = {"postings": SAMPLE_POSTINGS}
    print(f"Server: {args.base_url} | postings: {len(SAMPLE_POSTINGS)} | tasks: {tasks}\n")

    timings = []
    for task in tasks:
        path = ROUTES[task]
        print(f"=== {task}  (POST {path}) ===")
        status, payload, elapsed = call(args.base_url, path, body)
        if args.time:
            print(f"  took {elapsed:.3f}s")
            timings.append((task, elapsed))
        if isinstance(payload, (dict, list)):
            print(f"  status {status}")
            print(json.dumps(payload, indent=2))
        else:
            print(f"  status {status}: {payload}")
        print()

    if args.time and timings:
        total = sum(t for _, t in timings)
        print("=== timing summary ===")
        for task, elapsed in timings:
            print(f"  {task:<12} {elapsed:.3f}s")
        print(f"  {'total':<12} {total:.3f}s")


if __name__ == "__main__":
    main()
