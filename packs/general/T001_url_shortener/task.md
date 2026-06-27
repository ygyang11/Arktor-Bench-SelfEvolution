---
id: T001_url_shortener
name: URL Shortener Service
labels:
  domain: software_engineering
  subdomain: web_service
  model_capability: [code, tool_use]
  harness_focus: [instructions, tools]
  complexity: long_horizon
  multimodal: false
  online: false
workspace_files: []
---

## Prompt

Implement an in-memory HTTP service for URL shortening and link analytics, using only the
Python standard library.

### Startup
- The service is started with `python app.py` and may be organized across multiple modules.
- It reads the listening port from the `PORT` environment variable, serves on `127.0.0.1:$PORT`,
  and becomes ready within a few seconds.
- All state is held in memory; the service uses no database or disk persistence.

### Data model
Each link has four fields:
1. `code` — the short code identifying the link.
2. `url` — the original target URL.
3. `hits` — the number of successful redirects served, starting at 0.
4. `created` — a monotonic integer assigned 1, 2, 3, … in order of creation.

### Endpoints

1. **`POST /shorten`** — create a short link. The request body is a JSON object with a required
   string field `url` and an optional string field `alias`.
   - A URL is valid only if it begins with `http://` or `https://` and has a non-empty host.
   - Without an alias, the code is the first 7 characters of the base62 encoding (alphabet
     `0-9A-Za-z`) of the big-endian integer of `sha256(url.encode("utf-8")).digest()`.
     Idempotency applies to this deterministic-code link only: if it does not yet exist, create
     it and return 201; otherwise return 200 with the existing code.
   - With an alias, a separate link is created under that alias, so a single URL may have several
     links, each with its own `hits`. The alias must match `^[A-Za-z0-9_-]{3,32}$` and must not be
     a reserved word (`api`, `shorten`). The code namespace is global: an alias is unavailable if
     any existing link already uses that code, whether it was created as an alias or as a
     deterministic code. An available alias returns 201; an alias already bound to this same URL
     returns 200; an alias already bound to a different URL, or colliding with an existing code,
     returns 409; a malformed or reserved alias returns 400.
   - A malformed JSON body, a missing `url`, or an invalid URL returns 400 with a JSON object
     containing a string field `error`.
   - A successful response is a JSON object with fields `code`, `short_url`, and `url`, where
     `short_url` is `http://127.0.0.1:$PORT/<code>`.

2. **`GET /<code>`** — resolve a short link.
   - If `code` exists, return 302 with the `Location` header set to the original URL exactly,
     preserving any query string or fragment, and increment that link's `hits`.
   - Otherwise return 404.

3. **`GET /api/stats/<code>`** — report usage.
   - If `code` exists, return 200 with a JSON object with fields `code`, `url`, `hits`, and
     `created`.
   - Otherwise return 404.

4. **`GET /api/links`** — list links. Query parameters: `page`, `page_size`, `sort`.
   - `page` is 1-based and defaults to 1. `page_size` defaults to 20; a value above 100 is
     clamped to 100.
   - `sort` is one of `created`, `-created`, `hits`, or `-hits` (default `created`); a leading
     `-` denotes descending order, and ties are broken by `created` ascending.
   - The response is a JSON object with fields `items` (a list of objects with fields `code`,
     `url`, `hits`, `created`), `page`, `page_size`, and `total`.
   - A page beyond the last returns an empty `items` list with status 200.

5. **`DELETE /api/links/<code>`** — delete a link.
   - If `code` exists, return 204 with no body, after which the code is unknown (a subsequent
     redirect or stats request returns 404).
   - Otherwise return 404.

6. **`GET /api/top`** — report the most-used links. The query parameter `n` defaults to 5.
   - Return 200 with a JSON object with field `items`, a list of the top `n` links (objects with
     fields `code`, `url`, `hits`, `created`) by `hits` descending, ties broken by `created`
     ascending. If `n` exceeds the number of links, return all of them.

### General rules
1. The path `/shorten` and every path under `/api/` are reserved and are never interpreted as
   short codes; consequently `api` and `shorten` cannot be aliases. A path is resolved as a
   `<code>` only if it is not reserved.
2. A request whose path matches a defined endpoint but uses an unsupported method returns 405
   (for example, `GET /shorten`); a request whose path matches no defined endpoint returns 404.
3. For `page`, `page_size`, and `n`, a non-integer or non-positive value returns 400, and an
   unsupported `sort` value returns 400. (A `page_size` above 100 is clamped to 100, not rejected.)
4. Every JSON response sets `Content-Type: application/json`.
5. A link's `hits` is incremented only on a successful 302 redirect.

## Auto Checks

- {id: shorten_valid_201,          desc: "POST a valid URL creates the deterministic link and returns 201 with code/short_url/url", weight: 2}
- {id: code_deterministic,         desc: "deterministic codes equal base62(sha256(url bytes))[:7] for unseen URLs", weight: 2}
- {id: shorten_idempotent_200,     desc: "re-shortening without an alias returns 200 with the existing deterministic code", weight: 2}
- {id: alias_create_and_idempotent, desc: "a valid alias creates a separate link (201); the same URL+alias again returns 200", weight: 2}
- {id: alias_collision_409,        desc: "an alias on a different URL or matching an existing code returns 409", weight: 1}
- {id: alias_rejected_400,         desc: "a malformed or reserved alias returns 400", weight: 1}
- {id: redirect_and_hit_counting,  desc: "a known code returns 302 with the exact Location; hits increments only on 302 and is unchanged by an unknown lookup", weight: 3}
- {id: stats_known,                desc: "stats for a known code returns code/url/hits/created", weight: 1}
- {id: list_pagination,            desc: "page/page_size return the correct slice and envelope", weight: 2}
- {id: list_boundaries,            desc: "a past-end page is empty with 200 and page_size above 100 is clamped to 100", weight: 2}
- {id: list_sort_orders,           desc: "created/hits ascending and descending with created tie-breaks", weight: 1}
- {id: invalid_query_400,          desc: "non-positive/non-integer page, page_size, n and an unknown sort return 400", weight: 1}
- {id: delete_then_gone,           desc: "DELETE returns 204 and the code is then absent from redirect, stats, list, and top", weight: 2}
- {id: unknown_code_404,           desc: "an unknown code on redirect, stats, and delete returns 404", weight: 1}
- {id: top_n_ranking,              desc: "top returns links by hits desc with created tie-breaks, default n=5, and n-overflow returns all", weight: 2}
- {id: bad_request_matrix,         desc: "malformed JSON, missing url, invalid url, and non-string url/alias return 400", weight: 2}
- {id: method_and_reserved_paths,  desc: "reserved paths are not code lookups; an unsupported method returns 405 and an unrelated path returns 404", weight: 2}
- {id: json_content_type,          desc: "every JSON response sets Content-Type: application/json", weight: 1}

## Judge Rubric

- id: error_handling
  desc: "Quality and consistency of error responses: every error caught (no uncaught exceptions, 500s, or stack traces), one JSON error shape applied uniformly across all endpoints, and no internal implementation detail leaked in any body"
  weight: 7
  levels:
    - {score: 0.0,  desc: "At least one error path returns an uncaught exception, a 500, or a stack trace"}
    - {score: 0.25, desc: "Every error is caught, but at least one returns an empty or non-JSON body"}
    - {score: 0.5,  desc: "Errors return JSON bodies, but the shape varies across endpoints, or a body leaks internal detail (a raw exception message or stack frame)"}
    - {score: 0.75, desc: "One consistent JSON error shape with no leaked internals on the input-validation and not-found paths, but the unsupported-method (405) or malformed-body path still deviates"}
    - {score: 1.0,  desc: "That one shape holds on every error path — unknown-resource (404), unsupported-method (405), and malformed-body cases included"}
- id: code_organization
  desc: "Separation of responsibilities — routing, request/response handling, link storage, code generation, and listing/analytics — regardless of whether the solution is one file or several modules"
  weight: 4
  levels:
    - {score: 0.0, desc: "A single function or block handles routing, storage, code generation, formatting, and analytics together"}
    - {score: 0.3, desc: "Each endpoint has its own handler, but storage, code generation, and formatting are duplicated or reimplemented inside the handlers"}
    - {score: 0.6, desc: "Shared helpers exist, yet at least one core responsibility is still duplicated or inlined across multiple handlers"}
    - {score: 1.0, desc: "Each core responsibility lives behind a dedicated unit and handlers delegate to them; an incidental inline step in a handler is fine"}
