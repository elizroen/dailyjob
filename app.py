# app.py
import os, re, json, sqlite3, hashlib, time
from datetime import datetime, timedelta
from urllib.parse import urlencode
import requests, yaml, pandas as pd
from bs4 import BeautifulSoup
from dateutil import parser as dtparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CONFIG_PATH = os.getenv("JOBREC_CONFIG", "config.yml")
DEFAULT_TIMEOUT = 25

def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dirs(cfg):
    os.makedirs(os.path.dirname(cfg["cache"]["sqlite_path"]), exist_ok=True)
    os.makedirs(cfg["output"]["out_dir"], exist_ok=True)

# ----------------------- Caching / seen jobs -----------------------
def db_conn(sqlite_path):
    conn = sqlite3.connect(sqlite_path)
    conn.execute("""CREATE TABLE IF NOT EXISTS seen_jobs (
        id TEXT PRIMARY KEY,
        url TEXT,
        title TEXT,
        company TEXT,
        source TEXT,
        first_seen TEXT
    )""")
    return conn

def seen_add(conn, job):
    h = job_hash(job)
    conn.execute("INSERT OR IGNORE INTO seen_jobs (id,url,title,company,source,first_seen) VALUES (?,?,?,?,?,?)",
                 (h, job.get("url",""), job.get("title",""), job.get("company",""), job.get("source",""), datetime.utcnow().isoformat()))
    conn.commit()

def is_seen(conn, job):
    h = job_hash(job)
    cur = conn.execute("SELECT 1 FROM seen_jobs WHERE id=?", (h,))
    return cur.fetchone() is not None

def job_hash(job):
    s = (job.get("url","") + "|" + job.get("title","") + "|" + job.get("company",""))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# ----------------------- Helpers -----------------------
def textify(html_or_text):
    if not html_or_text: return ""
    if "<" in html_or_text and ">" in html_or_text:
        return BeautifulSoup(html_or_text, "html.parser").get_text(" ", strip=True)
    return html_or_text

def normalize(job):
    for k in ["title","company","location","url","description","source","posted_at"]:
        job.setdefault(k, "")
    job["description"] = textify(job["description"])
    return job

def keep_recent(job, days=45):
    if not job.get("posted_at"):
        return True
    try:
        dt = dtparse.parse(job["posted_at"])
        return (datetime.utcnow() - dt) <= timedelta(days=days)
    except Exception:
        return True

def any_in(text, arr):
    return any(a.lower() in (text or "").lower() for a in arr)

def matches_filters(job, prof):
    t = " ".join([job.get("title",""), job.get("description","")]).lower()
    loc = (job.get("location") or "")

    if prof.get("must_have_keywords") and not all(k.lower() in t for k in prof["must_have_keywords"]):
        return False
    if prof.get("block_keywords") and any(k.lower() in t for k in prof["block_keywords"]):
        return False

    if prof.get("seniority", {}).get("exclude") and any_in(job.get("title",""), prof["seniority"]["exclude"]):
        return False

    # locations
    if prof.get("locations", {}).get("exclude") and any_in(loc, prof["locations"]["exclude"]):
        return False
    inc_locs = prof.get("locations", {}).get("include", [])
    if inc_locs:
        if loc:
            if not any_in(loc, inc_locs) and "remote" not in loc.lower():
                return False
        # if no location present, allow (will be scored)

    return True

def backoff_sleep(try_i):
    time.sleep(min(1.5 * (try_i + 1), 6))

# ----------------------- Sources: Core ATS -----------------------
def fetch_lever(company):
    url = f"https://jobs.lever.co/{company}?mode=json"
    for i in range(2):
        try:
            r = requests.get(url, timeout=DEFAULT_TIMEOUT)
            r.raise_for_status()
            out = []
            for x in r.json():
                out.append(normalize({
                    "title": x.get("text") or x.get("title",""),
                    "company": company,
                    "location": (x.get("categories") or {}).get("location",""),
                    "url": x.get("hostedUrl") or x.get("applyUrl") or x.get("url",""),
                    "description": x.get("descriptionPlain") or x.get("description",""),
                    "source": "lever",
                    "posted_at": x.get("createdAt") or x.get("createdAtISO","")
                }))
            return out
        except Exception:
            backoff_sleep(i)
    return []

def fetch_greenhouse(company):
    url = f"https://boards-api.greenhouse.io/v1/boards/{company}/jobs?content=true"
    for i in range(2):
        try:
            r = requests.get(url, timeout=DEFAULT_TIMEOUT)
            r.raise_for_status()
            js = r.json().get("jobs", [])
            out = []
            for x in js:
                loc = (x.get("location") or {}).get("name","")
                out.append(normalize({
                    "title": x.get("title",""),
                    "company": company,
                    "location": loc,
                    "url": x.get("absolute_url") or "",
                    "description": x.get("content") or "",
                    "source": "greenhouse",
                    "posted_at": x.get("updated_at") or ""
                }))
            return out
        except Exception:
            backoff_sleep(i)
    return []

def fetch_ashby(org_id):
    url = "https://api.ashbyhq.com/api/public/jobs"
    payload = {"organizationId": org_id}
    for i in range(2):
        try:
            r = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
            r.raise_for_status()
            js = r.json().get("jobs", [])
            out = []
            for x in js:
                loc = ", ".join(filter(None, [
                    x.get("location",""),
                    "Remote" if x.get("remote") else ""
                ])).strip(", ")
                out.append(normalize({
                    "title": x.get("title",""),
                    "company": (x.get("organization") or {}).get("name",""),
                    "location": loc,
                    "url": x.get("jobUrl",""),
                    "description": (x.get("descriptionHtml") or x.get("descriptionPlain") or ""),
                    "source": "ashby",
                    "posted_at": x.get("createdAt") or ""
                }))
            return out
        except Exception:
            backoff_sleep(i)
    return []

# ----------------------- Sources: World Bank (SmartRecruiters) -----------------------
def fetch_smartrecruiters(org):
    url = f"https://api.smartrecruiters.com/v1/companies/{org}/postings"
    for i in range(2):
        try:
            r = requests.get(url, timeout=DEFAULT_TIMEOUT)
            if r.status_code != 200:
                backoff_sleep(i); continue
            js = r.json().get("content", [])
            out = []
            for x in js:
                out.append(normalize({
                    "title": x.get("name",""),
                    "company": org,
                    "location": ((x.get("location") or {}).get("city") or ""),
                    "url": (x.get("ref") or {}).get("jobAdUrl",""),
                    "description": (x.get("jobAd") or {}).get("sections",{}).get("jobDescription",""),
                    "source": "smartrecruiters",
                    "posted_at": x.get("createdOn","")
                }))
            return out
        except Exception:
            backoff_sleep(i)
    return []

# ----------------------- Sources: USAJOBS -----------------------
def fetch_usajobs(cfg, profile):
    usaj = cfg.get("sources", {}).get("usajobs", {})
    if not usaj or not usaj.get("enabled"): 
        return []
    email = usaj.get("email")
    api_key = usaj.get("api_key")
    if not email or not api_key:
        return []

    # Build a simple query using profile titles/skills and preferred locations (US/DC/Remote)
    keywords = list(set(profile.get("titles", []) + profile.get("skills", []) + profile.get("must_have_keywords", [])))
    kw = " OR ".join([f'"{k}"' if " " in k else k for k in keywords]) or "evaluation OR learning"
    locs = profile.get("locations", {}).get("include", []) or ["United States", "Remote", "Washington, DC"]

    headers = {
        "Host": "data.usajobs.gov",
        "User-Agent": email,
        "Authorization-Key": api_key,
    }

    out = []
    for loc in locs[:3]:  # keep it reasonable
        params = {
            "Keyword": kw,
            "LocationName": loc,
            "ResultsPerPage": 50,
            "WhoMayApply": "all"  # broaden; you can tweak
        }
        url = f"https://data.usajobs.gov/api/search?{urlencode(params)}"
        for i in range(2):
            try:
                r = requests.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)
                if r.status_code != 200:
                    backoff_sleep(i); continue
                data = r.json()
                items = data.get("SearchResult", {}).get("SearchResultItems", [])
                for it in items:
                    m = it.get("MatchedObjectDescriptor", {})
                    title = m.get("PositionTitle","")
                    org = m.get("OrganizationName","USAJOBS")
                    locs_txt = ", ".join([l.get("LocationName","") for l in m.get("PositionLocation", [])]) or ""
                    url_apply = m.get("PositionURI","")
                    desc = textify(m.get("UserArea",{}).get("Details",{}).get("JobSummary",""))
                    posted = m.get("PublicationStartDate","") or m.get("PublicationDate","")
                    out.append(normalize({
                        "title": title,
                        "company": org,
                        "location": locs_txt,
                        "url": url_apply,
                        "description": desc,
                        "source": "usajobs",
                        "posted_at": posted
                    }))
                break
            except Exception:
                backoff_sleep(i)
    return out

# ----------------------- Sources: Workday (Gates Foundation) -----------------------
def fetch_workday(tenant_shortname="gatesfoundation", career_site="GatesFoundation"):
    """
    Tries the public Workday CXS endpoint first. If unavailable, returns [].
    Some tenants require cookies/token—this function is best-effort.
    """
    # Common public JSON pattern:
    # https://<tenant>.myworkdayjobs.com/wday/cxs/<tenant>/<career_site>/jobs
    bases = [
        f"https://{tenant_shortname}.wd1.myworkdayjobs.com/wday/cxs/{tenant_shortname}/{career_site}/jobs",
        f"https://{tenant_shortname}.myworkdayjobs.com/wday/cxs/{tenant_shortname}/{career_site}/jobs",
    ]
    params = {"offset": 0, "limit": 100}
    for base in bases:
        for i in range(2):
            try:
                r = requests.get(base, params=params, timeout=DEFAULT_TIMEOUT)
                if r.status_code != 200:
                    backoff_sleep(i); continue
                js = r.json()
                jobs = js.get("jobPostings", []) or js.get("jobPostings", [])
                out = []
                for j in jobs:
                    url = j.get("externalPath","")
                    if url and not url.startswith("http"):
                        url = f"https://{tenant_shortname}.wd1.myworkdayjobs.com{url}"
                    out.append(normalize({
                        "title": j.get("title",""),
                        "company": "Gates Foundation",
                        "location": j.get("locationsText","") or j.get("location",""),
                        "url": url,
                        "description": textify((j.get("externalPostingDescription") or "") + " " + (j.get("qualificationsDescription") or "")),
                        "source": "workday",
                        "posted_at": j.get("postedOn","") or j.get("startDate","")
                    }))
                if out:
                    return out
            except Exception:
                backoff_sleep(i)
    return []

# ----------------------- Sources: Generic RSS (HigherEdJobs / Philanthropy) -----------------------
def fetch_rss(url, source="rss"):
    for i in range(2):
        try:
            r = requests.get(url, timeout=DEFAULT_TIMEOUT)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "xml")
            out = []
            for item in soup.find_all("item"):
                title = item.title.text if item.title else ""
                link = item.link.text if item.link else ""
                desc = item.description.text if item.description else ""
                pub = item.pubDate.text if item.pubDate else ""
                # Try to grab company/location from description if present
                company = ""
                location = ""
                out.append(normalize({
                    "title": title,
                    "company": company,
                    "location": location,
                    "url": link,
                    "description": desc,
                    "source": source,
                    "posted_at": pub
                }))
            return out
        except Exception:
            backoff_sleep(i)
    return []

# ----------------------- Scoring -----------------------
def build_profile_text(p):
    blocks = []
    blocks += p.get("titles", [])
    blocks += p.get("skills", [])
    blocks += p.get("must_have_keywords", [])
    blocks += p.get("nice_to_have_keywords", [])
    return ". ".join(blocks)

def score_jobs(jobs, profile_text):
    if not jobs: return []
    docs = [profile_text] + [ (j["title"] + " \n " + j["description"]) for j in jobs ]
    vec = TfidfVectorizer(stop_words="english", max_df=0.9)
    X = vec.fit_transform(docs)
    prof_vec = X[0:1]
    job_vecs = X[1:]
    sims = cosine_similarity(prof_vec, job_vecs).ravel()
    scored = []
    for j, s in zip(jobs, sims):
        boost = 0.0
        title = j["title"].lower()
        if any(t in title for t in ["director","head","lead","principal","senior"]):
            boost += 0.05
        if any(k in title for k in ["evaluation","learning","grants","customer success","policy"]):
            boost += 0.05
        scored.append((s + boost, j))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored

# ----------------------- Output -----------------------
def write_report(top, cfg):
    if not top: return None
    dt = datetime.now().strftime("%Y-%m-%d")
    out_dir = cfg["output"]["out_dir"]
    html_path = os.path.join(out_dir, f"jobs_{dt}.html")
    csv_log = cfg["output"]["csv_log"]

    rows, html_items = [], []
    for s, j in top:
        rows.append({
            "date": dt,
            "score": round(float(s), 4),
            "title": j["title"],
            "company": j.get("company",""),
            "location": j.get("location",""),
            "url": j["url"],
            "source": j["source"],
            "posted_at": j.get("posted_at","")
        })
        html_items.append(f"""
        <li>
          <a href="{j['url']}" target="_blank" rel="noopener noreferrer"><strong>{j['title']}</strong></a>
          — {j.get('company','')} | {j.get('location','')} | <em>{j['source']}</em><br/>
          <small>Score: {round(float(s),3)} | Posted: {j.get('posted_at','')}</small>
        </li>
        """)

    os.makedirs(os.path.dirname(csv_log), exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_log, mode="a", header=not os.path.exists(csv_log), index=False)

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Job picks {dt}</title></head>
<body>
<h2>Top job picks for {dt}</h2>
<ol>
{''.join(html_items)}
</ol>
<p style="color:#666">Generated by job-recommender.</p>
</body></html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path

# ----------------------- Main -----------------------
def main():
    cfg = load_config()
    ensure_dirs(cfg)
    conn = db_conn(cfg["cache"]["sqlite_path"])

    prof = cfg["profile"]
    profile_text = build_profile_text(prof)

    jobs = []

    # Core ATS sources
    for c in cfg["sources"].get("lever_companies", []):
        jobs += fetch_lever(c)
    for c in cfg["sources"].get("greenhouse_companies", []):
        jobs += fetch_greenhouse(c)
    for a in cfg["sources"].get("ashby_orgs", []):
        if isinstance(a, dict) and a.get("orgId"):
            jobs += fetch_ashby(a["orgId"])
        elif isinstance(a, str):
            jobs += fetch_ashby(a)

    # World Bank (SmartRecruiters)
    for s in cfg["sources"].get("smartrecruiters_orgs", []):
        org = s["org"] if isinstance(s, dict) else s
        jobs += fetch_smartrecruiters(org)

    # USAJOBS
    jobs += fetch_usajobs(cfg, prof)

    # Gates (Workday)
    wd = cfg["sources"].get("workday", {})
    if wd and wd.get("enabled"):
        t = wd.get("tenant_shortname","gatesfoundation")
        site = wd.get("career_site","GatesFoundation")
        jobs += fetch_workday(t, site)

    # HigherEd & Philanthropy RSS
    for url in cfg["sources"].get("highered_rss", []):
        jobs += fetch_rss(url, source="highered")
    for url in cfg["sources"].get("philanthropy_rss", []):
        jobs += fetch_rss(url, source="philanthropy")

    # Clean, filter, dedupe
    cleaned = []
    seen_urls = set()
    for j in jobs:
        j = normalize(j)
        if not j.get("url") or not j.get("title"): 
            continue
        if not keep_recent(j, days=cfg.get("recency_days", 45)): 
            continue
        if not matches_filters(j, prof):
            continue
        if j["url"] in seen_urls:
            continue
        cleaned.append(j)
        seen_urls.add(j["url"])

    # Fresh only (not previously recommended)
    fresh = [j for j in cleaned if not is_seen(conn, j)]

    # Score and pick top K
    scored = score_jobs(fresh, profile_text)
    top_k = cfg["output"].get("top_k", 3)
    top = scored[:top_k]

    for _, j in top:
        seen_add(conn, j)

    path = write_report(top, cfg)

    if top:
        print(f"Recommended {len(top)} jobs. Report: {path}")
        for s, j in top:
            print(f"- {j['title']} @ {j.get('company','')} ({j.get('location','')}) [{j['source']}] {j['url']} | score={round(float(s),3)}")
    else:
        print("No new matches today.")

if __name__ == "__main__":
    main()
