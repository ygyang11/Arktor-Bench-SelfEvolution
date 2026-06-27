"""In-sandbox HTTP grader driver for T001_url_shortener."""
import hashlib
import http.client
import json
import os
import socket
import subprocess
import sys
import time
from collections import namedtuple

_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def expected_code(url):
    n = int.from_bytes(hashlib.sha256(url.encode("utf-8")).digest(), "big")
    out = []
    while n:
        n, r = divmod(n, 62)
        out.append(_ALPHABET[r])
    return "".join(reversed(out))[:7]


def _free_port():
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


class Server:
    def __init__(self):
        self.port = _free_port()
        self.proc = subprocess.Popen(
            [sys.executable, "app.py"],
            env={**os.environ, "PORT": str(self.port)},
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def ready(self, timeout=8.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.proc.poll() is not None:
                return False
            try:
                with socket.create_connection(("127.0.0.1", self.port), 0.25):
                    return True
            except OSError:
                time.sleep(0.1)
        return False

    def finish(self):
        try:
            self.proc.terminate()
            _, err = self.proc.communicate(timeout=3)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
            err = b""
        lines = (err or b"").decode("utf-8", "replace").strip().splitlines()
        return lines[-1][:160] if lines else ""


Resp = namedtuple("Resp", "status headers data ctype")


def req(port, method, path, body=None, headers=None, raw=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    if raw is not None:
        payload = raw.encode("utf-8") if isinstance(raw, str) else raw
    elif body is not None:
        payload = json.dumps(body).encode("utf-8")
    else:
        payload = None
    h = dict(headers or {})
    if payload is not None and not any(k.lower() == "content-type" for k in h):
        h["Content-Type"] = "application/json"
    conn.request(method, path, body=payload, headers=h)
    resp = conn.getresponse()
    raw_body = resp.read()
    rh = {k.lower(): v for k, v in resp.getheaders()}
    conn.close()
    try:
        data = json.loads(raw_body) if raw_body else None
    except ValueError:
        data = None
    return Resp(resp.status, rh, data, rh.get("content-type", ""))


def grade(cases):
    fails = [m for ok, m in cases if not ok]
    n = len(cases) or 1
    return round((len(cases) - len(fails)) / n, 2), "; ".join(fails)


def _code(r):
    return r.data.get("code") if isinstance(r.data, dict) else None


# --- scenarios ---------------------------------------------------------------

def s_shorten_valid_201(port):
    u = "https://docs.example.org/guide/intro?ref=a"
    r = req(port, "POST", "/shorten", {"url": u})
    b = r.data if isinstance(r.data, dict) else {}
    code = b.get("code")
    return grade([
        (r.status == 201, f"POST a valid URL returned {r.status}, expected 201"),
        (isinstance(code, str) and bool(code), "response has no string `code`"),
        (b.get("url") == u, "response `url` does not echo the submitted URL"),
        (isinstance(code, str) and b.get("short_url") == f"http://127.0.0.1:{port}/{code}",
         "`short_url` is not http://127.0.0.1:$PORT/<code>"),
    ])


def s_code_deterministic(port):
    cases = []
    for u in ("http://a.example/one", "https://b.example/two?q=2", "https://c.example/x#f"):
        got = _code(req(port, "POST", "/shorten", {"url": u}))
        exp = expected_code(u)
        cases.append((got == exp, f"code was {got!r}, expected the deterministic {exp!r}"))
    return grade(cases)


def s_shorten_idempotent_200(port):
    u = "https://idem.example/path?a=b"
    r1 = req(port, "POST", "/shorten", {"url": u})
    if r1.status != 201 or not _code(r1):
        return 0.0, "precondition failed: the first shorten did not return 201 with a code"
    r2 = req(port, "POST", "/shorten", {"url": u})
    return grade([
        (r2.status == 200, f"re-shortening returned {r2.status}, expected 200"),
        (_code(r2) == _code(r1), "re-shortening returned a different code"),
    ])


def s_alias_create_and_idempotent(port):
    u, a = "https://alias.example/x", "my-alias-1"
    r1 = req(port, "POST", "/shorten", {"url": u, "alias": a})
    r2 = req(port, "POST", "/shorten", {"url": u, "alias": a})
    r3 = req(port, "POST", "/shorten", {"url": u})         # same URL, no alias -> a separate det link
    det = expected_code(u)
    return grade([
        (r1.status == 201, f"a free alias returned {r1.status}, expected 201"),
        (_code(r1) == a, "the created link's code is not the requested alias"),
        (r2.status == 200, f"re-creating the same URL+alias returned {r2.status}, expected 200"),
        (r3.status == 201 and _code(r3) == det,
         "shortening the same URL without an alias did not create a separate deterministic link"),
        (req(port, "GET", "/" + a).status == 302 and req(port, "GET", "/" + det).status == 302,
         "the alias link and the deterministic link do not both resolve independently"),
    ])


def s_alias_collision_409(port):
    cases = []
    req(port, "POST", "/shorten", {"url": "https://one.example/a", "alias": "shared-x"})
    r = req(port, "POST", "/shorten", {"url": "https://two.example/b", "alias": "shared-x"})
    cases.append((r.status == 409,
                  f"an alias bound to a different URL returned {r.status}, expected 409"))
    code = _code(req(port, "POST", "/shorten", {"url": "https://det.example/c"}))
    if not code:
        cases.append((False, "precondition: could not obtain a deterministic code to collide with"))
    else:
        r2 = req(port, "POST", "/shorten", {"url": "https://det.example/d", "alias": code})
        cases.append((r2.status == 409,
                      f"an alias equal to an existing code returned {r2.status}, expected 409"))
    return grade(cases)


def s_alias_rejected_400(port):
    u = "https://rej.example/x"
    bad = {
        "too-short alias": "ab",
        "too-long alias": "z" * 33,
        "alias with a space": "has space",
        "alias with a bad char": "no!bang",
        "reserved alias api": "api",
        "reserved alias shorten": "shorten",
    }
    return grade([
        (req(port, "POST", "/shorten", {"url": u, "alias": a}).status == 400,
         f"{label} did not return 400")
        for label, a in bad.items()
    ])


def s_redirect_and_hit_counting(port):
    u = "https://hit.example/a/b?q=1&r=2#frag"
    code = _code(req(port, "POST", "/shorten", {"url": u}))
    if not code:
        return 0.0, "precondition failed: could not create a link to redirect"
    rr = req(port, "GET", "/" + code)
    for _ in range(2):
        req(port, "GET", "/" + code)               # 3 successful redirects total
    req(port, "GET", "/unknown-zzz")               # an unknown lookup must not change hits
    hits = req(port, "GET", "/api/stats/" + code).data
    hits = hits.get("hits") if isinstance(hits, dict) else None
    return grade([
        (rr.status == 302, f"redirect returned {rr.status}, expected 302"),
        (rr.headers.get("location") == u, "redirect Location is not the exact original URL"),
        (hits == 3, f"hits is {hits} after 3 redirects and one unknown lookup, expected 3"),
    ])


def s_stats_known(port):
    u = "https://stats.example/x"
    code = _code(req(port, "POST", "/shorten", {"url": u}))
    if not code:
        return 0.0, "precondition failed: could not create a link"
    r = req(port, "GET", "/api/stats/" + code)
    b = r.data if isinstance(r.data, dict) else {}
    return grade([
        (r.status == 200, f"stats returned {r.status}, expected 200"),
        (b.get("code") == code, "stats `code` is wrong"),
        (b.get("url") == u, "stats `url` is wrong"),
        (b.get("hits") == 0, f"stats `hits` is {b.get('hits')!r} before any redirect, expected 0"),
        (isinstance(b.get("created"), int) and b.get("created", 0) >= 1,
         "stats `created` is not a positive integer"),
    ])


def s_list_pagination(port):
    created = [_code(req(port, "POST", "/shorten", {"url": f"https://list.example/{i}"}))
               for i in range(5)]
    if not all(created):
        return 0.0, "precondition failed: could not create the links to paginate"
    r = req(port, "GET", "/api/links?page=1&page_size=2")
    b = r.data if isinstance(r.data, dict) else {}
    r2 = req(port, "GET", "/api/links?page=2&page_size=2")
    b2 = r2.data if isinstance(r2.data, dict) else {}
    page1 = [it.get("code") for it in b.get("items", [])] if isinstance(b.get("items"), list) else None
    page2 = [it.get("code") for it in b2.get("items", [])] if isinstance(b2.get("items"), list) else None
    return grade([
        (r.status == 200, f"list returned {r.status}, expected 200"),
        (page1 == created[0:2], "page 1 is not the first two links in creation order"),
        (b.get("page") == 1 and b.get("page_size") == 2 and b.get("total") == 5,
         "page/page_size/total fields are wrong"),
        (r2.status == 200 and page2 == created[2:4],
         "page 2 is not the next two links in creation order"),
    ])


def s_list_boundaries(port):
    for i in range(3):
        req(port, "POST", "/shorten", {"url": f"https://bound.example/{i}"})
    r = req(port, "GET", "/api/links?page=99&page_size=10")
    items = r.data.get("items") if isinstance(r.data, dict) else None
    r2 = req(port, "GET", "/api/links?page=1&page_size=1000")
    psz = r2.data.get("page_size") if isinstance(r2.data, dict) else None
    return grade([
        (r.status == 200 and items == [],
         f"a past-end page returned status {r.status} / items {items!r}, expected 200 with []"),
        (psz == 100, f"page_size above 100 was {psz!r}, expected clamp to 100"),
    ])


def s_list_sort_orders(port):
    codes = [_code(req(port, "POST", "/shorten", {"url": f"https://sort.example/{i}"}))
             for i in range(4)]
    if not all(codes):
        return 0.0, "precondition failed: could not create the links for sorting"
    plan = {0: 1, 1: 3, 2: 2, 3: 1}                 # codes[0] and codes[3] tie at 1 hit
    for i, h in plan.items():
        for _ in range(h):
            req(port, "GET", "/" + codes[i])

    def order(sort):
        items = req(port, "GET", f"/api/links?sort={sort}&page_size=100").data
        items = items.get("items") if isinstance(items, dict) else []
        return [it.get("code") for it in items if isinstance(it, dict) and it.get("code") in codes]

    created_asc = list(codes)                       # creation order: 0, 1, 2, 3
    return grade([
        (order("created") == created_asc, "sort=created is not in creation order"),
        (order("-created") == created_asc[::-1], "sort=-created is not reverse creation order"),
        (order("-hits") == [codes[1], codes[2], codes[0], codes[3]],
         "sort=-hits is not hits-desc with a created-asc tie-break"),
        (order("hits") == [codes[0], codes[3], codes[2], codes[1]],
         "sort=hits is not hits-asc with a created-asc tie-break"),
    ])


def s_invalid_query_400(port):
    cases = {
        "page=0": "/api/links?page=0",
        "page=-1": "/api/links?page=-1",
        "page=abc": "/api/links?page=abc",
        "page_size=0": "/api/links?page_size=0",
        "sort=unknown": "/api/links?sort=whatever",
        "top n=0": "/api/top?n=0",
        "top n=abc": "/api/top?n=abc",
    }
    return grade([(req(port, "GET", p).status == 400, f"{label} did not return 400")
                  for label, p in cases.items()])


def s_delete_then_gone(port):
    code = _code(req(port, "POST", "/shorten", {"url": "https://del.example/x"}))
    if not code:
        return 0.0, "precondition failed: could not create a link to delete"
    req(port, "GET", "/" + code)                    # one hit so it would appear in top
    d = req(port, "DELETE", "/api/links/" + code)
    lst = req(port, "GET", "/api/links?page_size=100").data
    lcodes = {it.get("code") for it in lst.get("items", [])} if isinstance(lst, dict) else set()
    top = req(port, "GET", "/api/top?n=100").data
    tcodes = {it.get("code") for it in top.get("items", [])} if isinstance(top, dict) else set()
    return grade([
        (d.status == 204, f"DELETE returned {d.status}, expected 204"),
        (req(port, "GET", "/" + code).status == 404, "redirect after delete did not 404"),
        (req(port, "GET", "/api/stats/" + code).status == 404, "stats after delete did not 404"),
        (code not in lcodes, "the deleted code still appears in the list"),
        (code not in tcodes, "the deleted code still appears in top"),
    ])


def s_unknown_code_404(port):
    miss = "no-such-code"
    return grade([
        (req(port, "GET", "/" + miss).status == 404, "redirect for an unknown code did not 404"),
        (req(port, "GET", "/api/stats/" + miss).status == 404, "stats for an unknown code did not 404"),
        (req(port, "DELETE", "/api/links/" + miss).status == 404, "delete for an unknown code did not 404"),
    ])


def s_top_n_ranking(port):
    codes = [_code(req(port, "POST", "/shorten", {"url": f"https://top.example/{i}"}))
             for i in range(6)]
    if not all(codes):
        return 0.0, "precondition failed: could not create the links for top ranking"
    plan = {0: 5, 1: 1, 2: 4, 3: 4, 4: 2, 5: 3}     # codes[2], codes[3] tie at 4 hits
    for i, h in plan.items():
        for _ in range(h):
            req(port, "GET", "/" + codes[i])
    expected = [codes[0], codes[2], codes[3], codes[5], codes[4], codes[1]]   # hits desc, created tie

    def top(q):
        d = req(port, "GET", "/api/top" + q).data
        return [it.get("code") for it in d.get("items", [])] if isinstance(d, dict) else []

    return grade([
        (top("?n=3") == expected[:3], "top n=3 is not the top 3 by hits with a created tie-break"),
        (top("") == expected[:5], "default top is not the top 5 in exact order"),
        (top("?n=100") == expected, "top n=100 does not return all links in exact order"),
    ])


def s_bad_request_matrix(port):
    return grade([
        (req(port, "POST", "/shorten", raw="{not json").status == 400, "malformed JSON did not 400"),
        (req(port, "POST", "/shorten", {}).status == 400, "missing `url` did not 400"),
        (req(port, "POST", "/shorten", {"url": "ftp://x.example/a"}).status == 400,
         "a non-http scheme did not 400"),
        (req(port, "POST", "/shorten", {"url": "not-a-url"}).status == 400,
         "a URL without a scheme did not 400"),
        (req(port, "POST", "/shorten", {"url": "http://"}).status == 400,
         "a URL with an empty host did not 400"),
        (req(port, "POST", "/shorten", {"url": 123}).status == 400, "a non-string `url` did not 400"),
        (req(port, "POST", "/shorten", {"url": "https://ok.example/a", "alias": 5}).status == 400,
         "a non-string `alias` did not 400"),
    ])


def s_method_and_reserved_paths(port):
    return grade([
        (req(port, "GET", "/shorten").status == 405, "GET /shorten was not 405"),
        (req(port, "POST", "/api/links").status == 405, "POST /api/links was not 405"),
        (req(port, "DELETE", "/api/stats/anything").status == 405, "DELETE /api/stats/.. was not 405"),
        (req(port, "GET", "/api/does-not-exist").status == 404, "an unknown /api path was not 404"),
    ])


def s_json_content_type(port):
    r = req(port, "POST", "/shorten", {"url": "https://ct.example/a"})
    cases = [("application/json" in r.ctype, f"201 response Content-Type was {r.ctype!r}")]
    code = _code(r)
    if code:
        rs = req(port, "GET", "/api/stats/" + code)
        cases.append(("application/json" in rs.ctype, f"stats Content-Type was {rs.ctype!r}"))
    rb = req(port, "POST", "/shorten", {})
    cases.append(("application/json" in rb.ctype, f"400 error Content-Type was {rb.ctype!r}"))
    return grade(cases)


SCENARIOS = {
    "shorten_valid_201": s_shorten_valid_201,
    "code_deterministic": s_code_deterministic,
    "shorten_idempotent_200": s_shorten_idempotent_200,
    "alias_create_and_idempotent": s_alias_create_and_idempotent,
    "alias_collision_409": s_alias_collision_409,
    "alias_rejected_400": s_alias_rejected_400,
    "redirect_and_hit_counting": s_redirect_and_hit_counting,
    "stats_known": s_stats_known,
    "list_pagination": s_list_pagination,
    "list_boundaries": s_list_boundaries,
    "list_sort_orders": s_list_sort_orders,
    "invalid_query_400": s_invalid_query_400,
    "delete_then_gone": s_delete_then_gone,
    "unknown_code_404": s_unknown_code_404,
    "top_n_ranking": s_top_n_ranking,
    "bad_request_matrix": s_bad_request_matrix,
    "method_and_reserved_paths": s_method_and_reserved_paths,
    "json_content_type": s_json_content_type,
}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = SCENARIOS.get(name)
    if fn is None:
        print(json.dumps({"score": 0.0, "msg": f"unknown scenario {name!r}"}))
        return
    if not os.path.exists("app.py"):
        print(json.dumps({"score": 0.0, "msg": "app.py was not created"}))
        return
    server = Server()
    if not server.ready():
        tail = server.finish()
        extra = f" (last stderr: {tail})" if tail else ""
        print(json.dumps(
            {"score": 0.0, "msg": f"the service did not start and listen on $PORT within 8s{extra}"}))
        return
    try:
        score, msg = fn(server.port)
    except Exception as e:  # noqa: BLE001
        score, msg = 0.0, f"the service produced an unusable response ({type(e).__name__})"
    finally:
        server.finish()
    print(json.dumps({"score": round(float(score), 2), "msg": msg}))


if __name__ == "__main__":
    main()
