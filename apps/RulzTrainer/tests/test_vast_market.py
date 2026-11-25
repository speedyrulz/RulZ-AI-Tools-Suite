"""scripts/vast/rulz_market.py: filters to search bodies, offers, the account's burn, templates, and the rent / stop /
destroy rules with the records that keep a rent from being lost or doubled. A fake opener stands in for vast.ai:
nothing here touches the network, spends money or uses a GPU."""
import json
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "vast"))

import rulz_market as mk  # noqa: E402
import rulz_remote as remote  # noqa: E402

KEY = "KEY-SECRET-0123456789"
ENV_SECRET = "HF_TOKEN_SECRET_VALUE_abc"
PUBKEY = "ssh-ed25519 AAAAC3Nza-test-key me@pc"


# ---------------------------------------------------------------- the fake vast.ai


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def offer_row(i=100, **kw):
    row = {"id": i, "ask_contract_id": i, "machine_id": 5, "host_id": 6, "gpu_name": "RTX 4090", "num_gpus": 1,
           "gpu_ram": 24564, "gpu_total_ram": 24564, "dph_total": 0.5, "dph_base": 0.42, "storage_total_cost": 0.08,
           "min_bid": 0.2, "dlperf": 40.0, "dlperf_per_dphtotal": 80.0, "flops_per_dphtotal": 160.0,
           "total_flops": 80.0, "reliability2": 0.99, "inet_up": 800, "inet_down": 900, "inet_up_cost": 0.01,
           "inet_down_cost": 0.02, "cpu_name": "EPYC", "cpu_cores": 32, "cpu_cores_effective": 16,
           "cpu_ram": 64000, "cpu_ghz": 3.1, "disk_bw": 2000, "disk_space": 500, "disk_name": "NVMe",
           "pcie_bw": 12, "pci_gen": 4, "gpu_mem_bw": 1000, "bw_nvlink": 0, "cuda_max_good": 12.9,
           "driver_version": "570.1", "os_version": "22.04", "geolocation": "Texas, US", "verification": "verified",
           "hosting_type": 1, "static_ip": True, "direct_port_count": 20, "duration": 864000, "mobo_name": "X",
           "rentable": True}
    row.update(kw)
    return row


def inst_row(i=1, **kw):
    row = {"id": i, "label": f"label-{i}", "actual_status": "running", "intended_status": "running",
           "gpu_name": "RTX 5090", "num_gpus": 1, "gpu_ram": 32000, "dph_total": 0.8, "storage_total_cost": 0.1,
           "disk_space": 200, "image_uuid": "vastai/pytorch:cuda-12.8.1-auto", "extra_env": [], "min_bid": 0.3,
           "start_date": 1700000000.0}
    row.update(kw)
    return row


class Vast:
    """The opener. Routes are answered from attributes the test sets; every call is recorded."""

    def __init__(self, home=None):
        self.calls = []
        self.home = home
        self.user = {"id": 7, "credit": 10.0, "balance": 0}
        self.offers = [offer_row(100)]
        self.hide = 0  # find answers that leave the offer out before one shows it
        self.instances = {}
        self.templates = []
        self.ssh = [{"public_key": PUBKEY}]
        self.rent_answer = (200, json.dumps({"success": True, "new_contract": 555}))
        self.rent_effect = None
        self.put_answer = (200, json.dumps({"success": True}))
        self.statuses = None  # per call of GET /instances/{id}/, a list of actual_status values
        self.intent_seen = None
        self.fail = None

    def __call__(self, method, url, headers, data, timeout):
        parts = urlsplit(url)
        path, query = parts.path, parse_qs(parts.query)
        body = json.loads(data) if data else None
        self.calls.append((method, path, body, query))
        assert headers["Authorization"] == f"Bearer {KEY}"
        if self.fail:
            exc = self.fail
            if callable(exc):
                exc = exc(method, path)
            if exc:
                raise exc
        m = re.fullmatch(r"/api/v0/(\w+)/(\d+)?/?", path)
        if path == "/api/v0/users/current/":
            return 200, json.dumps(self.user)
        if path == "/api/v0/bundles/":
            rows = self.offers
            if "ask_contract_id" in body:
                if self.hide > 0:
                    self.hide -= 1
                    return 200, json.dumps({"offers": []})
                rows = [r for r in rows if r["id"] == body["ask_contract_id"]["eq"]]
            return 200, json.dumps({"offers": rows})
        if path == "/api/v0/ssh/":
            return (200, json.dumps(self.ssh)) if method == "GET" else (200, json.dumps({"success": True}))
        if path == "/api/v0/template/":
            return 200, json.dumps({"templates": self.templates})
        if path == "/api/v1/instances/":
            rows = [dict(r) for r in self.instances.values()]
            start = int(query.get("after_token", ["0"])[0])
            page = rows[start:start + 25]
            out = {"instances": page}
            if start + 25 < len(rows):
                out["next_token"] = str(start + 25)
            return 200, json.dumps(out)
        if method == "PUT" and m and m.group(1) == "asks":
            if self.home:
                self.intent_seen = json.loads((self.home / "rentals.json").read_text(encoding="utf-8"))
            if self.rent_effect:
                self.rent_effect(body)
            return self.rent_answer
        mi = re.fullmatch(r"/api/v0/instances/(\d+)/", path)
        if mi:
            iid = int(mi.group(1))
            if method == "GET":
                if iid not in self.instances:
                    return 404, json.dumps({"msg": "not found"})
                row = dict(self.instances[iid])
                if self.statuses:
                    row["actual_status"] = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
                    row["intended_status"] = "running" if row["actual_status"] == "running" else "stopped"
                return 200, json.dumps({"instances": row})
            return self.put_answer
        if path.startswith("/api/v0/instances/bid_price/"):
            return self.put_answer
        raise AssertionError(f"unexpected call {method} {path}")

    def of(self, method, pattern):
        return [c for c in self.calls if c[0] == method and re.search(pattern, c[1])]


@pytest.fixture
def world(tmp_path):
    clock = Clock()
    vast = Vast(tmp_path / "home")
    ssh = tmp_path / "ssh"
    ssh.mkdir()
    (ssh / "id_ed25519.pub").write_text(PUBKEY + "\n", encoding="utf-8")
    api = mk.Api(opener=vast, key=KEY, sleep=clock.sleep)
    market = mk.Market(home=tmp_path / "home", api=api, now=clock.now, sleep=clock.sleep, which=lambda n: "/bin/ssh",
                       ssh_dir=ssh)
    market.vast, market.clock, market.ssh_path = vast, clock, ssh
    return market


def an_offer(**kw):
    return mk.parse_offer(offer_row(**kw))


# ---------------------------------------------------------------- the client


def test_errors_are_rewritten_and_never_hold_the_key():
    def boom(*a):
        raise OSError(f"Connection to https://x failed with Bearer {KEY}")

    api = mk.Api(opener=boom, key=KEY, sleep=lambda s: None)
    with pytest.raises(mk.MarketError) as e:
        api.get("/api/v0/users/current/")
    assert KEY not in e.value.message and e.value.kind == "network" and not e.value.definite

    def timeout(*a):
        raise TimeoutError(KEY)

    with pytest.raises(mk.MarketError) as e:
        mk.Api(opener=timeout, key=KEY).get("/x")
    assert "in time" in e.value.message and KEY not in e.value.message

    echo = mk.Api(opener=lambda *a: (400, json.dumps({"msg": f"bad token {KEY}"})), key=KEY)
    with pytest.raises(mk.MarketError) as e:
        echo.get("/x")
    assert KEY not in e.value.message and "***" in e.value.message and e.value.definite


@pytest.mark.parametrize("status,kind,definite", [(401, "auth", False), (403, "auth", False), (500, "server", False),
                                                  (503, "server", False)])
def test_status_kinds(status, kind, definite):
    api = mk.Api(opener=lambda *a: (status, ""), key=KEY)
    with pytest.raises(mk.MarketError) as e:
        api.get("/x")
    assert (e.value.kind, e.value.definite) == (kind, definite)


def test_a_4xx_is_definite_only_with_a_json_body():
    api = mk.Api(opener=lambda *a: (404, "<html>"), key=KEY)
    with pytest.raises(mk.MarketError) as e:
        api.get("/x")
    assert e.value.definite is False
    api = mk.Api(opener=lambda *a: (422, json.dumps({"error": "nope"})), key=KEY)
    with pytest.raises(mk.MarketError) as e:
        api.get("/x")
    assert e.value.definite is True and "nope" in e.value.message


def test_no_key_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(mk, "api_key", lambda: None)
    with pytest.raises(mk.MarketError) as e:
        mk.Api(opener=lambda *a: (200, "{}")).get("/x")
    assert e.value.kind == "no-key" and e.value.definite


def test_reads_retry_a_429_with_backoff_then_give_up():
    clock, answers = Clock(), [(429, "")] * 3 + [(200, "{}")]
    api = mk.Api(opener=lambda *a: answers.pop(0), key=KEY, sleep=clock.sleep)
    assert api.get("/x") == {}
    assert clock.t - 1_000_000.0 == 1 + 2 + 4
    clock2, count = Clock(), []
    api = mk.Api(opener=lambda *a: (count.append(1), (429, ""))[1], key=KEY, sleep=clock2.sleep)
    with pytest.raises(mk.MarketError) as e:
        api.get("/x")
    assert e.value.kind == "rate" and len(count) == 5 and clock2.t - 1_000_000.0 == 15


def test_api_key_order(monkeypatch, tmp_path):
    monkeypatch.setattr(mk.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    assert mk.api_key() is None
    (tmp_path / ".vast_api_key").write_text("old\n", encoding="utf-8")
    assert mk.api_key() == "old"
    (tmp_path / ".config" / "vastai").mkdir(parents=True)
    (tmp_path / ".config" / "vastai" / "vast_api_key").write_text("cli\n", encoding="utf-8")
    assert mk.api_key() == "cli"
    monkeypatch.setenv("VAST_API_KEY", "env")
    assert mk.api_key() == "env"


def test_instances_read_every_page_by_25_with_next_token(world):
    world.vast.instances = {i: inst_row(i) for i in range(1, 61)}
    assert len(world.instances()) == 60
    pages = world.vast.of("GET", "/api/v1/instances/")
    assert len(pages) == 3
    assert pages[0][3]["limit"] == ["25"] and pages[1][3]["after_token"] == ["25"]


def test_a_page_loop_is_an_error_not_a_short_list(world):
    world.vast.instances = {1: inst_row(1)}
    api = mk.Api(opener=lambda *a: (200, json.dumps({"instances": [], "next_token": "same"})), key=KEY)
    with pytest.raises(mk.MarketError):
        api.instances_raw()


def test_find_offer_asks_by_ask_contract_id_twice(world):
    world.vast.hide = 1
    row = world.api.find_offer(100, {"limit": 10})
    assert row["id"] == 100
    sent = world.vast.of("POST", "bundles")
    assert len(sent) == 2 and sent[0][2]["ask_contract_id"] == {"eq": 100} and "id" not in sent[0][2]
    world.vast.hide = 2
    assert world.api.find_offer(100, {"limit": 10}) is None


# ---------------------------------------------------------------- filters -> body


def test_default_filters_and_body():
    f = mk.default_filters()
    assert f["type"] == "ask" and f["num_gpus"] == "1" and f["disk"] == 200 and f["reliability"] == 98
    assert f["gpu_ram"] == [24, None] and f["inet_down"] == [500, None] and f["sort"] == "dlperf_per_dollar"
    assert f["unverified"] is False
    body = mk.search_body(f)
    assert body["limit"] == 1000 and body["type"] == "ask" and body["external"] == {"eq": False}
    assert body["allocated_storage"] == 200 and body["disk_space"] == {"gte": 200}
    assert body["order"] == [["dlperf_per_dphtotal", "desc"]]
    assert body["num_gpus"] == {"eq": 1} and body["reliability2"] == {"gte": 0.98}
    assert body["gpu_ram"] == {"gte": 23520} and body["inet_down"] == {"gte": 500}
    assert body["verified"] == {"eq": True} and body["rentable"] == {"eq": True}
    assert body["cuda_max_good"] == {"gte": 12.8} and body["direct_port_count"] == {"gte": 4}
    assert body["cpu_arch"] == {"in": ["amd64"]} and body["image_tag"] == {"eq": "vastai/pytorch:cuda-12.8.1-auto"}


# every control other than the client-side one and the sort changes the body when it is set
CHANGES = {
    "type": "bid", "num_gpus": "4", "gpu_name": ["RTX 4090"], "region": "eu", "countries": ["DE"], "disk": 300,
    "reliability": 90, "duration": 2, "secure_cloud": True, "unverified": True, "incompatible": True,
    "unavailable": True, "static_ip": True, "min_cuda": 12.9, "min_driver": "535.86.05", "ubuntu_version": "22.04",
    "dph": [0.1, 2], "up_cost": [None, 20], "down_cost": [1, None], "gpu_count": [2, 4], "total_flops": 50,
    "gpu_ram": [16, 48], "gpu_total_ram": [32, 96], "gpu_mem_bw": [500, None], "pcie_bw": [8, None],
    "bw_nvlink": [100, None], "dlperf": [10, 90], "cpu_cores": [8, 64], "cpu_ram": [32, 256], "cpu_ghz": [2.5, None],
    "disk_bw": [500, None], "inet_up": [100, None], "inet_down": [1000, 5000], "direct_port_count": [8, None],
    "pci_gen": [3, None], "gpu_max_power": [200, 700], "has_avx": True, "gpu_frac": [0.5, None],
    "machine_id": "42", "host_id": "9",
}


@pytest.mark.parametrize("key", [f.key for f in mk.FILTERS if f.key not in ("tflops_price", "sort")])
def test_every_filter_changes_the_body(key):
    base = mk.default_filters()
    # the incompatible-machines rule adds fields a control might hide behind: compare with it on and off
    for incompatible in (False, True):
        base["incompatible"] = incompatible
        changed = dict(base, **{key: CHANGES[key]})
        if key == "incompatible":
            continue
        assert mk.search_body(changed) != mk.search_body(base), key
    if key == "incompatible":
        assert mk.search_body(dict(base, incompatible=True)) != mk.search_body(dict(base, incompatible=False))


def test_filter_details():
    f = dict(mk.default_filters(), **{**CHANGES, "num_gpus": "any"})
    b = mk.search_body(f)
    assert b["type"] == "bid" and b["num_gpus"] == {"gte": 2, "lte": 4}
    assert b["gpu_name"] == {"in": ["RTX 4090"]} and b["geolocation"] == {"in": ["DE"]}  # countries beat region
    assert b["allocated_storage"] == 300 and b["disk_space"] == {"gte": 300}
    assert b["reliability2"] == {"gte": 0.9} and b["duration"] == {"gte": 172800}
    assert b["datacenter"] == {"eq": True} and "verified" not in b and "rentable" not in b
    assert b["static_ip"] == {"eq": True} and b["cuda_max_good"] == {"gte": 12.9}
    assert b["driver_version"] == {"gte": 535086005} and b["ubuntu_version"] == {"eq": "22.04"}
    assert b["dph_total"] == {"gte": 0.1, "lte": 2}
    assert b["inet_up_cost"] == {"lte": 0.02} and b["inet_down_cost"] == {"gte": 0.001}  # $/TB -> $/GB
    assert b["total_flops"] == {"gte": 50}
    assert b["gpu_ram"] == {"gte": 15680, "lte": 49152}  # N GB min is N*1000*0.98, max N*1024
    assert b["gpu_total_ram"] == {"gte": 31360, "lte": 98304}
    assert b["cpu_ram"] == {"gte": 31360, "lte": 262144} and b["cpu_cores_effective"] == {"gte": 8, "lte": 64}
    assert b["direct_port_count"] == {"gte": 8} and b["has_avx"] == {"eq": True}
    assert b["machine_id"] == {"eq": 42} and b["host_id"] == {"eq": 9}
    assert "cpu_arch" not in b and "image_tag" not in b  # incompatible machines were switched on


def test_num_gpus_choices():
    def gpus(choice, rng=(None, None)):
        return mk.search_body(dict(mk.default_filters(), num_gpus=choice, gpu_count=list(rng))).get("num_gpus")

    assert gpus("any") is None
    assert gpus("1") == {"eq": 1} and gpus("8") == {"eq": 8}
    assert gpus("9+") == {"gte": 9}
    assert gpus("any", (2, 4)) == {"gte": 2, "lte": 4}
    assert gpus("4", (2, 8)) == {"eq": 4}


def test_regions_and_countries():
    b = mk.search_body(dict(mk.default_filters(), region="north_america"))
    assert "geolocation" not in b  # not a region: falls back to the default (Planet Earth)
    b = mk.search_body(dict(mk.default_filters(), region="na"))
    assert set(b["geolocation"]["in"]) >= {"US", "CA"} and "DE" not in b["geolocation"]["in"]
    assert "geolocation" not in mk.search_body(mk.default_filters())
    assert set(mk.REGIONS) == {"na", "sa", "eu", "as", "af", "oc"}
    assert mk.COUNTRY_NAMES["DE"] == "Germany" and "DE" in mk.REGIONS["eu"]


def test_template_sets_the_cuda_and_image():
    template = {"image": "vastai/pytorch", "tag": "cuda-12.4.1-auto"}
    b = mk.search_body(mk.default_filters(), template)
    assert b["cuda_max_good"] == {"gte": 12.4} and b["image_tag"] == {"eq": "vastai/pytorch:cuda-12.4.1-auto"}


@pytest.mark.parametrize("sort,order", [("auto", ["score", "desc"]), ("dlperf", ["dlperf", "desc"]),
                                        ("price_up", ["dph_total", "asc"]), ("price_down", ["dph_total", "desc"]),
                                        ("tflops", ["total_flops", "desc"]), ("vram", ["gpu_total_ram", "desc"]),
                                        ("tflops_per_dollar", ["flops_per_dphtotal", "desc"]),
                                        ("dlperf_per_dollar", ["dlperf_per_dphtotal", "desc"])])
def test_sorts(sort, order):
    assert mk.search_body(dict(mk.default_filters(), sort=sort))["order"] == [order]
    assert len(mk.SORTS) == 9


def test_clean_filters_forces_values_into_range():
    c = mk.clean_filters({"bogus": 1, "disk": 999999, "reliability": -5, "gpu_ram": [40, 10], "type": "nope",
                          "num_gpus": 2, "gpu_name": ["RTX 4090", "RTX 4090", "bad;name", 7], "countries": ["de", "xx1"],
                          "dph": "x", "secure_cloud": "true", "machine_id": "12a", "min_driver": "5;5", "sort": 3,
                          "cpu_cores": [None, 99999], "duration": ""})
    assert "bogus" not in c and set(c) == {f.key for f in mk.FILTERS}
    assert c["disk"] == 4000 and c["reliability"] == 0 and c["gpu_ram"] == [10, 40] and c["type"] == "ask"
    assert c["num_gpus"] == "2" and c["gpu_name"] == ["RTX 4090"] and c["countries"] == ["DE"]
    assert c["dph"] == [None, None] and c["secure_cloud"] is True and c["machine_id"] == ""
    assert c["min_driver"] == "" and c["sort"] == "dlperf_per_dollar" and c["cpu_cores"] == [None, None]
    assert c["duration"] is None
    assert mk.clean_filters(None) == mk.default_filters() == mk.clean_filters(mk.default_filters())
    assert mk.clean_filters({"disk": None})["disk"] == 200  # a required value never goes blank


def test_filter_schema_carries_what_a_screen_needs():
    keys = [f.key for f in mk.FILTERS]
    assert len(keys) == len(set(keys))
    assert {f.group for f in mk.FILTERS} <= set(mk.GROUPS)
    for f in mk.FILTERS:
        d = f.to_dict()
        assert {"key", "group", "label", "kind", "unit", "lo", "hi", "step", "default", "hint"} <= set(d)
        assert f.kind in ("range", "min", "switch", "choice", "multi", "text")
        if f.kind in ("choice", "multi") and f.options:
            assert all(len(o) == 2 for o in f.options)
    assert json.dumps([f.to_dict() for f in mk.FILTERS])


def test_client_side_price_per_tflops_and_vram_sort():
    cheap = an_offer(i=1, dph_total=0.4, total_flops=80)  # 0.005 per TFLOPS
    dear = an_offer(i=2, dph_total=2.0, total_flops=80)  # 0.025
    none = an_offer(i=3, dph_total=0.1, total_flops=None)
    f = dict(mk.default_filters(), tflops_price=[None, 0.01])
    assert [o.id for o in mk.client_side([cheap, dear, none], f)] == [1]
    f = dict(mk.default_filters(), tflops_price=[0.01, None])
    assert [o.id for o in mk.client_side([cheap, dear, none], f)] == [2]
    a = an_offer(i=1, dph_total=1.0, gpu_total_ram=24000)
    b = an_offer(i=2, dph_total=0.5, gpu_total_ram=24000)
    c = an_offer(i=3, dph_total=3.0, gpu_ram=24000, gpu_total_ram=96000)
    out = mk.client_side([a, b, c], dict(mk.default_filters(), sort="vram_per_dollar"))
    assert [o.id for o in out] == [2, 3, 1]


def test_search_parses_and_drops_unusable_rows(world):
    world.vast.offers = [offer_row(1), {"id": 2}, offer_row(3, dph_total=None), offer_row(4, num_gpus=0),
                         offer_row(5, tflops=1)]
    offers = mk.search(world.api, mk.default_filters())
    assert [o.id for o in offers] == [1, 5]
    assert world.vast.of("POST", "bundles")[0][2]["limit"] == 1000


# ---------------------------------------------------------------- offers


def test_parse_offer_reads_the_fields():
    o = mk.parse_offer(offer_row(7))
    assert (o.id, o.machine_id, o.host_id, o.gpu_name, o.num_gpus) == (7, 5, 6, "RTX 4090", 1)
    assert o.gpu_ram_mb == 24564 and o.dph_total == 0.5 and o.dph_base == 0.42 and o.storage_dph == 0.08
    assert o.reliability == 0.99 and o.verified and o.datacenter and o.static_ip and o.ports == 20
    assert o.cpu_cores == 16 and o.cpu_cores_total == 32 and o.cuda == 12.9 and o.country == "US"
    assert o.duration_days == pytest.approx(10) and o.type == "ask" and o.rentable
    assert o.tflops == 80 and o.tflops_per_dollar == 160 and o.dlperf_per_dollar == 80
    assert json.dumps(o.to_dict())


def test_parse_offer_with_nulls():
    raw = {k: None for k in offer_row()}
    raw.update(id=9, dph_total=1.0, num_gpus=2, gpu_ram=12000)
    o = mk.parse_offer(raw)
    assert o.id == 9 and o.gpu_total_ram_mb == 24000 and o.dlperf is None and o.reliability is None
    assert o.tflops is None and o.tflops_per_dollar is None and o.duration_days is None
    assert o.verified is False and o.datacenter is False and o.country == "" and o.rentable is True


@pytest.mark.parametrize("raw", [None, "x", {}, {"id": 1}, {"id": 1, "dph_total": -1, "num_gpus": 1, "gpu_ram": 1},
                                 {"id": 1, "dph_total": 1, "num_gpus": 1, "gpu_ram": 0}])
def test_parse_offer_unusable(raw):
    assert mk.parse_offer(raw) is None


def test_parse_offer_verification_and_type():
    assert mk.parse_offer(offer_row(verification="unverified")).verified is False
    assert mk.parse_offer(offer_row(verification="deverified")).verified is False
    o = mk.parse_offer(offer_row(dph_base=0.2, min_bid=0.2, dph_total=0.28), "bid")
    assert o.type == "bid" and o.min_bid == o.dph_base == 0.2


# ---------------------------------------------------------------- account


def test_account_burn_and_hours_left(world):
    v = world.vast
    v.user = {"id": 7, "credit": 10.0, "balance": 2.0}
    v.instances = {
        1: inst_row(1, dph_total=0.8, storage_total_cost=0.1),
        2: inst_row(2, actual_status="exited", intended_status="stopped", dph_total=0.5, storage_total_cost=0.05),
        3: inst_row(3, actual_status="loading", intended_status="running", dph_total=0.4, num_gpus=2),
        4: inst_row(4, actual_status=None, intended_status="running", dph_total=0.2),
        5: inst_row(5, actual_status="exited", intended_status="stopped", storage_total_cost=None, storage_cost=73.0,
                    disk_space=100, dph_total=0.9),
        6: inst_row(6, actual_status="exited", intended_status="running", dph_total=0.3),  # asked to run: billed
    }
    a = world.account()
    by = {i["id"]: i for i in a["instances"]}
    assert a["credit"] == 12.0
    assert by[1]["per_hour"] == 0.8 and by[1]["billing"] is True
    assert by[2]["per_hour"] == 0.05 and by[2]["billing"] is False
    assert by[3]["per_hour"] == 0.4 and by[3]["billing"] and by[3]["status"] == "loading"
    assert by[4]["billing"] and by[4]["status"] == "loading"
    assert by[5]["per_hour"] == pytest.approx(73.0 * 100 / 730)  # $/GB/month * GB / 730
    assert by[6]["billing"] is False and by[6]["per_hour"] == 0.1 * 0 + 0.1
    burn = a["burn"]
    assert burn["per_hour"] == pytest.approx(0.8 + 0.05 + 0.4 + 0.2 + 10.0 + 0.1)
    assert (burn["running"], burn["stopped"], burn["gpus"]) == (3, 3, 4)
    assert burn["disks"] == 1100
    assert a["hours_left"] == pytest.approx(12.0 / burn["per_hour"])
    assert a["empty_at"] == pytest.approx(world.clock.now() + a["hours_left"] * 3600)
    assert all(i["ours"] is False for i in a["instances"])


def test_account_nothing_billed_and_no_credit(world):
    world.vast.user = {"id": 7}
    a = world.account()
    assert a["credit"] is None and a["hours_left"] is None and a["empty_at"] is None
    assert a["burn"]["per_hour"] == 0 and a["instances"] == []
    assert mk.lasts(5, 0) is None and mk.lasts(None, 1) is None and mk.lasts(-3, 1) == 0
    assert mk.lasts(10, 2.5) == 4.0


def test_a_stopped_instance_with_no_price_info_costs_nothing_not_its_running_price(world):
    world.vast.instances = {1: inst_row(1, actual_status="exited", intended_status="stopped", storage_total_cost=None,
                                        storage_cost=None, dph_total=0.9)}
    assert world.account()["burn"]["per_hour"] == 0


def test_instances_know_which_are_rulz_and_ours(world):
    world.vast.instances = {
        1: inst_row(1, extra_env=[["RULZ_REPO", "https://example/x.git"], ["-p 22:22", "1"]]),
        2: inst_row(2, extra_env=[["PROVISIONING_SCRIPT", "https://raw/x/rulz/app/provision.sh"]]),
        3: inst_row(3, extra_env=[["OTHER", "1"]]),
        4: inst_row(4, image_uuid="me/rulztrainer:latest", extra_env=None),
    }
    world.market = world
    got = {i["id"]: i for i in world.instances()}
    assert [got[i]["is_rulz"] for i in (1, 2, 3, 4)] == [True, True, False, True]
    assert all("extra_env" not in i for i in got.values())


# ---------------------------------------------------------------- templates


def template_row(hash_id, created, env, **kw):
    row = {"id": created, "hash_id": hash_id, "name": f"t{hash_id}", "image": "vastai/pytorch",
           "tag": "cuda-12.8.1-auto", "recommended_disk_space": 150.0, "created_at": created, "env": env,
           "ssh_direct": True}
    row.update(kw)
    return row


CURRENT_ENV = (f"-p 22:22 -e HF_TOKEN={ENV_SECRET} -e RULZ_REPO=https://x/y.git "
               "-e PROVISIONING_SCRIPT=https://raw.githubusercontent.com/o/r/app/rulz-trainer/scripts/vast/provision.sh "
               "-e PORTAL_CONFIG='a:1:2:/:A|b:3:4:/:B'")


def test_templates_keep_only_rulz_ones_newest_first_without_env(world):
    world.vast.templates = [
        template_row("old", 100, "-e PROVISIONING_SCRIPT=https://raw/o/rulz/app/old-branch/provision.sh "
                               f"-e SECRET={ENV_SECRET}"),
        template_row("other", 150, f"-e FOO=bar -e KEY={ENV_SECRET}"),
        template_row("new", 300, CURRENT_ENV),
        template_row("dup", 200, CURRENT_ENV),
        template_row("new", 300, CURRENT_ENV),
        template_row("img", 50, "-p 1:1", image="me/rulztrainer"),
        template_row("rulzenv", 10, "-e RULZ_BRANCH=x"),
        template_row("", 400, CURRENT_ENV),
    ]
    items = mk.templates(world.api)
    assert [t["hash_id"] for t in items] == ["new", "dup", "old", "img", "rulzenv"]
    assert [t["current"] for t in items] == [True, True, False, False, False]
    assert set(items[0]) == {"id", "hash_id", "name", "image", "tag", "disk", "created", "current", "cuda"}
    assert items[0]["disk"] == 150.0 and items[0]["cuda"] == 12.8
    assert ENV_SECRET not in json.dumps(items) and "env" not in items[0]
    call = world.vast.of("GET", "template")[0]
    assert json.loads(call[3]["select_filters"][0]) == {"creator_id": {"eq": 7}}
    assert mk.default_template(items)["hash_id"] == "new"
    assert mk.default_template(items[2:])["builtin"] is True


def test_no_user_id_no_templates(world):
    world.vast.user = {}
    assert mk.templates(world.api) == []


# ---------------------------------------------------------------- renting


def rent_it(world, **kw):
    offer = kw.pop("offer", None) or an_offer(i=100)
    return world.rent(offer, kw.pop("filters", mk.default_filters()), kw.pop("template", None), **kw)


def test_rent_happy_path_with_builtin_template(world):
    world.vast.home = world.home
    world.vast.instances = {555: inst_row(555, label="x", dph_total=0.5)}
    rental = rent_it(world, disk=200, send_id="s1", label_note="for a test")
    put = world.vast.of("PUT", "asks")
    assert len(put) == 1 and put[0][1] == "/api/v0/asks/100/"
    body = put[0][2]
    assert re.fullmatch(r"rulz-trainer-[0-9a-f]{8}", body["label"]) and body["disk"] == 200
    assert body["target_state"] == "running" and body["cancel_unavail"] is True and "template_hash_id" not in body
    assert body["image"] == "vastai/pytorch:cuda-12.8.1-auto" and body["runtype"] == "jupyter_direc ssh_direc ssh_proxy"
    assert body["onstart"] == "/opt/instance-tools/bin/entrypoint.sh" and "price" not in body
    assert body["env"]["RULZ_BRANCH"] == "app/rulz-trainer"
    # the intent was on disk, before the PUT, with everything needed to find the rent again
    seen = world.vast.intent_seen["intents"]
    assert len(seen) == 1 and seen[0]["label"] == body["label"] and seen[0]["offer"] == 100
    assert seen[0]["price"] == 0.5 and seen[0]["send"] == "s1" and seen[0]["state"] == "sending"
    assert rental["id"] == 555 and rental["label"] == body["label"] and rental["price"] == 0.5
    assert rental["send"] == "s1" and rental["note"] == "for a test" and rental["price_over"] is False
    assert world.intents() == [] and [r["id"] for r in world.rentals()] == [555]
    assert world.rentals()[0]["real_price"] == 0.5


def test_rent_with_a_template_sends_its_hash_and_not_its_environment(world):
    rent_it(world, template={"hash_id": "abc", "image": "i", "tag": "cuda-12.8.1-auto"}, disk=150)
    body = world.vast.of("PUT", "asks")[0][2]
    assert body == {"template_hash_id": "abc", "disk": 150, "label": body["label"], "target_state": "running",
                    "cancel_unavail": True}


def test_the_offer_is_looked_up_again_with_the_same_storage(world):
    rent_it(world, disk=250)
    lookup = world.vast.of("POST", "bundles")[0][2]
    assert lookup["ask_contract_id"] == {"eq": 100} and lookup["allocated_storage"] == 250
    assert world.vast.calls.index(world.vast.of("POST", "bundles")[0]) < \
        world.vast.calls.index(world.vast.of("PUT", "asks")[0])


def test_a_clear_refusal_drops_the_intent_and_says_nothing_was_rented(world):
    world.vast.rent_answer = (400, json.dumps({"msg": "offer taken"}))
    with pytest.raises(mk.MarketError) as e:
        rent_it(world)
    assert e.value.definite is True and e.value.kind == "client"
    assert world.intents() == [] and world.rentals() == []
    world.vast.rent_answer = (200, json.dumps({"success": False, "msg": "no longer available"}))
    with pytest.raises(mk.MarketError) as e:
        rent_it(world)
    assert e.value.definite and "no longer available" in e.value.message and world.intents() == []
    assert len(world.vast.of("PUT", "asks")) == 2  # each click is its own rent: never an automatic retry


@pytest.mark.parametrize("answer", [(500, ""), (200, "not json"), (200, json.dumps({"success": True})),
                                    (502, json.dumps({"msg": "bad gateway"})), (200, "[]")])
def test_an_unclear_answer_leaves_the_intent_checking(world, answer):
    world.vast.rent_answer = answer
    with pytest.raises(mk.MarketError) as e:
        rent_it(world)
    assert e.value.kind == "pending" and e.value.definite is False and e.value.data["label"]
    (intent,) = world.intents()
    assert intent["state"] == "checking" and intent["label"] == e.value.data["label"]
    assert len(world.vast.of("PUT", "asks")) == 1


def test_a_timeout_and_a_429_on_the_rent_are_never_retried(world):
    world.vast.fail = lambda m, p: TimeoutError() if m == "PUT" and "asks" in p else None
    with pytest.raises(mk.MarketError) as e:
        rent_it(world)
    assert e.value.kind == "pending" and len(world.vast.of("PUT", "asks")) == 1
    world2_intent = world.intents()[0]
    assert world2_intent["state"] == "checking"
    world.vast.fail = None
    world._drop_intent(world2_intent["label"])
    world.vast.rent_answer = (429, json.dumps({"msg": "slow down"}))
    with pytest.raises(mk.MarketError) as e:
        rent_it(world)
    assert e.value.kind == "rate" and e.value.definite and world.intents() == []
    assert len(world.vast.of("PUT", "asks")) == 2 and world.clock.t == 1_000_000.0  # no sleeping, no second try
    # a 429 or a 401 from something that is not vast.ai's JSON (a proxy page) may have come from anywhere
    for status in (429, 401, 403):
        world.vast.rent_answer = (status, "<html>")
        with pytest.raises(mk.MarketError) as e:
            rent_it(world)
        assert e.value.kind == "pending" and not e.value.definite
        world._drop_intent(e.value.data["label"])


def test_a_second_rent_is_refused_while_one_is_open(world):
    world.vast.rent_answer = (500, "")
    with pytest.raises(mk.MarketError):
        rent_it(world)
    world.vast.rent_answer = (200, json.dumps({"success": True, "new_contract": 556}))
    with pytest.raises(mk.MarketError) as e:
        rent_it(world)
    assert e.value.kind == "busy" and e.value.definite
    assert len(world.vast.of("PUT", "asks")) == 1


def lost_rent(world, **kw):
    world.vast.rent_answer = (500, "")
    with pytest.raises(mk.MarketError) as e:
        rent_it(world, **kw)
    return e.value.data["label"]


def test_settle_finds_the_label_and_adopts(world):
    label = lost_rent(world, send_id="s9")
    assert world.settle() == []  # the first look is at 5 s
    world.clock.sleep(6)
    world.vast.instances = {900: inst_row(900, label="someone-else"), 901: inst_row(901, label=label, dph_total=0.5)}
    (change,) = world.settle()
    assert change["state"] == "found" and change["label"] == label
    assert change["rental"]["id"] == 901 and change["rental"]["send"] == "s9"
    assert world.intents() == [] and [r["id"] for r in world.rentals()] == [901]


def test_settle_follows_the_schedule_then_goes_unresolved(world):
    label = lost_rent(world)
    looks = []
    orig = world.api.instances_raw
    world.api.instances_raw = lambda: (looks.append(world.clock.now()), orig())[1]
    t0 = world.clock.now()
    for _ in range(700):
        world.clock.sleep(1)
        world.settle()
    assert [round(t - t0) for t in looks] == list(mk.CHECK_OFFSETS)
    (intent,) = world.intents()
    assert intent["state"] == "unresolved" and label in intent["message"] and "10 minutes" in intent["message"]
    assert world.rentals() == []
    n = len(looks)
    world.clock.sleep(1000)
    world.settle()
    assert len(looks) == n  # nothing more on its own: the user clicks Check again or Dismiss


def test_a_failed_search_is_not_one_of_the_checks(world):
    lost_rent(world)
    world.clock.sleep(6)
    world.vast.fail = lambda m, p: OSError() if "/api/v1/" in p else None
    assert world.settle() == []
    assert world.intents()[0]["done"] == 0


def test_a_crash_leaves_a_sending_intent_that_settles_on_the_next_start(world, tmp_path):
    world.vast.home = world.home
    seen = {}

    def crash(body):
        seen["label"] = body["label"]
        raise KeyboardInterrupt  # the process dies in the PUT

    world.vast.rent_effect = crash
    with pytest.raises(KeyboardInterrupt):
        rent_it(world)
    assert world.intents()[0]["state"] == "sending"
    # a new process
    again = mk.Market(home=world.home, api=world.api, now=world.clock.now, sleep=world.clock.sleep,
                      which=lambda n: "ssh", ssh_dir=world.ssh_path)
    assert again.settle() == [] and again.intents()[0]["state"] == "checking"
    assert again.intents()[0]["since"] == world.clock.now()
    with pytest.raises(mk.MarketError) as e:
        again.rent(an_offer(i=100), mk.default_filters())
    assert e.value.kind == "busy"
    world.clock.sleep(6)
    world.vast.instances = {77: inst_row(77, label=seen["label"])}
    (change,) = again.settle()
    assert change["state"] == "found" and change["rental"]["id"] == 77


def test_check_again_and_dismiss(world):
    label = lost_rent(world)
    with pytest.raises(mk.MarketError) as e:  # while the checks run it cannot be dismissed
        world.dismiss_intent(label)
    assert e.value.kind == "busy"
    assert world.check_intent(label) == {"found": False, "rental": None}
    world.vast.instances = {5: inst_row(5, label=label)}
    got = world.check_intent(label)
    assert got["found"] and got["rental"]["id"] == 5 and world.intents() == []
    # unresolved -> dismissable
    world.vast.instances = {}
    label = lost_rent(world)
    for _ in range(700):
        world.clock.sleep(1)
        world.settle()
    world.check_intent(label)
    assert "Still no instance" in world.intents()[0]["message"]
    world.dismiss_intent(label)
    assert world.intents() == []


def test_a_manual_check_that_cannot_read_the_account_raises(world):
    label = lost_rent(world)
    world.vast.fail = lambda m, p: OSError() if "/api/v1/" in p else None
    with pytest.raises(mk.MarketError) as e:
        world.check_intent(label)
    assert "Could not read" in e.value.message and len(world.intents()) == 1


def test_price_recheck_refuses_a_rise_and_names_the_new_price(world):
    world.vast.offers = [offer_row(100, dph_total=0.6)]
    with pytest.raises(mk.MarketError) as e:
        rent_it(world, offer=an_offer(i=100, dph_total=0.5))
    assert e.value.kind == "price" and e.value.definite and "$0.600" in e.value.message
    assert e.value.data["new_price"] == 0.6 and world.vast.of("PUT", "asks") == [] and world.intents() == []


def test_price_within_half_a_percent_or_lower_is_rented(world):
    world.vast.offers = [offer_row(100, dph_total=0.502)]
    assert rent_it(world, offer=an_offer(i=100, dph_total=0.5))["id"] == 555
    world.vast.instances = {}
    world.vast.offers = [offer_row(100, dph_total=0.4)]
    world._drop_intent("none")
    assert rent_it(world, offer=an_offer(i=100, dph_total=0.5))["price"] == 0.5


def test_an_offer_that_is_gone_or_not_rentable_is_refused(world):
    world.vast.offers = []
    with pytest.raises(mk.MarketError) as e:
        rent_it(world)
    assert e.value.kind == "gone" and e.value.definite and world.vast.of("PUT", "asks") == []
    assert len(world.vast.of("POST", "bundles")) == 2  # asked twice before it counts as gone
    world.vast.offers = [offer_row(100, rentable=False)]
    with pytest.raises(mk.MarketError) as e:
        rent_it(world)
    assert e.value.kind == "gone"


def test_the_contract_price_read_back_is_reported_when_over_two_percent(world):
    world.vast.instances = {555: inst_row(555, dph_total=0.55)}
    rental = rent_it(world)
    assert rental["price_over"] is True and rental["real_price"] == 0.55 and "$0.550" in rental["note"]
    world.vast.instances = {555: inst_row(555, dph_total=0.505)}
    world.rentals()  # a within-2% price is not reported
    world._drop_intent("x")
    world.vast.rent_answer = (200, json.dumps({"success": True, "new_contract": 556}))
    world.vast.instances = {556: inst_row(556, dph_total=0.505)}
    assert rent_it(world)["price_over"] is False


def test_the_read_back_failing_does_not_lose_the_rental(world):
    world.vast.instances = {}  # GET answers 404: no real price known
    rental = rent_it(world)
    assert rental["id"] == 555 and rental["real_price"] is None and world.rentals()[0]["id"] == 555


def test_interruptible_rent_needs_a_bid_at_or_above_the_fresh_minimum(world):
    world.vast.offers = [offer_row(100, dph_base=0.2, min_bid=0.25, dph_total=0.33)]
    bid_offer = mk.parse_offer(world.vast.offers[0], "bid")
    f = dict(mk.default_filters(), type="bid")
    with pytest.raises(mk.MarketError) as e:
        world.rent(bid_offer, f)
    assert e.value.kind == "bid" and "bid" in e.value.message
    with pytest.raises(mk.MarketError) as e:
        world.rent(bid_offer, f, bid=0.2)
    assert e.value.kind == "bid" and e.value.data["min_bid"] == 0.25 and e.value.definite
    assert world.vast.of("PUT", "asks") == []
    world.vast.offers = [offer_row(100, dph_base=0.2, min_bid=0.25, dph_total=0.33)]
    world.vast.rent_answer = (200, json.dumps({"success": True, "new_contract": 600}))
    rental = world.rent(bid_offer, f, bid=0.3, template={"hash_id": "h"})
    body = world.vast.of("PUT", "asks")[0][2]
    assert body["price"] == 0.3 and rental["bid"] == 0.3 and rental["type"] == "bid"
    assert rental["price"] == pytest.approx(0.3 + 0.08)  # the shown total is bid + storage
    assert world.vast.of("POST", "bundles")[0][2]["type"] == "bid"


def test_a_bid_is_ignored_for_on_demand_and_reserved_is_refused(world):
    rent_it(world, bid=0.01)
    assert "price" not in world.vast.of("PUT", "asks")[0][2]
    with pytest.raises(mk.MarketError) as e:
        world.rent(mk.parse_offer(offer_row(100), "reserved"), mk.default_filters())
    assert e.value.kind == "reserved"


def test_ssh_precondition_before_the_rent(world):
    world.which = lambda n: None
    with pytest.raises(mk.MarketError) as e:
        rent_it(world)
    assert e.value.kind == "ssh" and e.value.definite and "ssh" in e.value.message
    world.which = lambda n: "ssh"
    for f in world.ssh_path.iterdir():
        f.unlink()
    with pytest.raises(mk.MarketError) as e:
        rent_it(world)
    assert e.value.kind == "ssh" and "public key" in e.value.message
    (world.ssh_path / "id_ed25519.pub").write_text(PUBKEY, encoding="utf-8")
    world.vast.ssh = [{"public_key": "ssh-ed25519 OTHER-KEY someone"}]
    with pytest.raises(mk.MarketError) as e:
        rent_it(world)
    assert e.value.kind == "ssh" and "account" in e.value.message
    assert world.vast.of("PUT", "asks") == [] and world.intents() == []
    world.vast.ssh = [{"public_key": PUBKEY.rsplit(" ", 1)[0] + " other-comment"}]  # same key, other comment
    assert rent_it(world)["id"] == 555


def test_ssh_key_state_and_add(world):
    assert world.ssh_key_state() == "on_account"
    world.vast.ssh = []
    assert world.ssh_key_state() == "not_on_account"
    assert world.add_ssh_key() == {"added": True}
    post = world.vast.of("POST", "/ssh/")
    assert post[0][2] == {"ssh_key": PUBKEY}
    world.vast.ssh = [PUBKEY]  # a plain string list also reads
    assert world.ssh_key_state() == "on_account" and world.add_ssh_key() == {"added": False}
    for f in world.ssh_path.iterdir():
        f.unlink()
    assert world.ssh_key_state() == "missing"
    with pytest.raises(mk.MarketError):
        world.add_ssh_key()


def test_an_invalid_offer_is_refused(world):
    with pytest.raises(mk.MarketError):
        world.rent({"id": 5})
    with pytest.raises(mk.MarketError):
        world.rent(mk.parse_offer(offer_row(dph_total=0.0)), mk.default_filters())


# ---------------------------------------------------------------- start, stop, destroy


def test_stop_rereads_the_instance_and_acts_on_that_id(world):
    world.vast.instances = {5: inst_row(5)}
    out = world.stop(5)
    assert out["id"] == 5 and out["ok"] is True
    calls = [c for c in world.vast.calls if "/instances/5/" in c[1]]
    assert [c[0] for c in calls] == ["GET", "PUT"] and calls[1][2] == {"state": "stopped"}
    assert [c for c in world.vast.calls if c[0] == "PUT" and "/instances/" in c[1] and "5" not in c[1]] == []


def test_stop_of_a_missing_instance_or_a_refusal(world):
    with pytest.raises(mk.MarketError) as e:
        world.stop(5)
    assert e.value.kind == "gone" and world.vast.of("PUT", "instances") == []
    world.vast.instances = {5: inst_row(5)}
    world.vast.put_answer = (200, json.dumps({"success": False, "msg": "cannot stop"}))
    with pytest.raises(mk.MarketError) as e:
        world.stop(5)
    assert e.value.definite and "cannot stop" in e.value.message
    with pytest.raises(mk.MarketError):
        world.stop("abc")
    with pytest.raises(mk.MarketError):
        world.stop(-1)


def test_start_only_a_stopped_instance(world):
    world.vast.instances = {5: inst_row(5, actual_status="exited", intended_status="stopped"), 6: inst_row(6)}
    assert world.start(5)["started"] is True
    assert world.vast.of("PUT", "/instances/5/")[0][2] == {"state": "running"}
    assert world.start(6)["started"] is False and world.vast.of("PUT", "/instances/6/") == []


def test_watch_stop_follows_until_vast_stops_reporting_running(world):
    world.vast.instances = {5: inst_row(5)}
    world.vast.statuses = ["running", "running", "exited"]
    t0 = world.clock.now()
    out = world.watch_stop(5)
    assert out["stopped"] is True and out["status"] == "exited"
    assert world.clock.now() - t0 == 360 and out["waited"] == 360  # looks every 2 minutes
    assert len(world.vast.of("GET", "/instances/5/")) == 3


def test_watch_stop_alerts_loudly_after_ten_minutes(world):
    world.vast.instances = {5: inst_row(5)}
    t0 = world.clock.now()
    out = world.watch_stop(5)
    assert out["stopped"] is False and "still reports it running" in out["alert"] and "billing" in out["alert"]
    assert world.clock.now() - t0 == 600 and len(world.vast.of("GET", "/instances/5/")) == 5


def test_watch_stop_survives_failed_looks_and_counts_a_gone_instance_as_stopped(world):
    world.vast.instances = {5: inst_row(5)}
    flaky = iter([OSError(), None, None])
    world.vast.fail = lambda m, p: next(flaky, None) if m == "GET" else None
    world.vast.statuses = ["exited"]
    assert world.watch_stop(5)["stopped"] is True
    world.vast.fail = None
    world.vast.instances = {}
    assert world.watch_stop(5) == {"stopped": True, "status": "gone", "waited": 120.0}


def test_stop_confirmed_is_stop_then_watch(world):
    world.vast.instances = {5: inst_row(5)}
    world.vast.statuses = ["exited"]
    out = world.stop_confirmed(5)
    assert out["ok"] and out["stopped"] and out["id"] == 5


def test_destroy_guards_and_records(world):
    world.vast.instances = {555: inst_row(555, label="other-label"), 9: inst_row(9, label="mine")}
    rental = rent_it(world)
    world.vast.instances[555]["label"] = "renamed"
    with pytest.raises(mk.MarketError) as e:  # the label is not the recorded one any more
        world.destroy(555, require_rental=True)
    assert e.value.kind == "refused" and world.vast.of("DELETE", "instances") == []
    with pytest.raises(mk.MarketError):  # not in rentals.json at all
        world.destroy(9, require_rental=True)
    assert world.vast.of("DELETE", "instances") == []
    world.vast.instances[555]["label"] = rental["label"]
    assert world.destroy(555, require_rental=True) == {"id": 555, "ok": True}
    assert [c[1] for c in world.vast.of("DELETE", "instances")] == ["/api/v0/instances/555/"]
    assert world.rentals()[0]["destroyed"]
    assert world.account()["instances"][0]["ours"] is False  # a destroyed rental is no longer ours
    # a click on any instance of the account, after its confirmation
    assert world.destroy(9)["ok"] and world.vast.of("DELETE", "instances")[-1][1] == "/api/v0/instances/9/"
    with pytest.raises(mk.MarketError) as e:
        world.destroy(12345)
    assert e.value.kind == "gone"


def test_destroy_checks_success(world):
    world.vast.instances = {9: inst_row(9)}
    calls = []
    world.api.call = lambda method, path, *a, **k: (calls.append((method, path)),
                                                     mk.Reply(200, {"success": False, "msg": "busy"}, True))[1]
    with pytest.raises(mk.MarketError) as e:
        world.api.delete_instance(9)
    assert e.value.definite and "busy" in e.value.message and calls == [("DELETE", "/api/v0/instances/9/")]


def test_change_bid(world):
    world.vast.instances = {5: inst_row(5, min_bid=0.3)}
    with pytest.raises(mk.MarketError) as e:
        world.change_bid(5, 0.2)
    assert e.value.kind == "bid" and world.vast.of("PUT", "bid_price") == []
    for bad in (0, -1, "x", None):
        with pytest.raises(mk.MarketError):
            world.change_bid(5, bad)
    assert world.change_bid(5, 0.35) == {"id": 5, "price": 0.35}
    (call,) = world.vast.of("PUT", "bid_price")
    assert call[1] == "/api/v0/instances/bid_price/5/" and call[2] == {"client_id": "me", "price": 0.35}


# ---------------------------------------------------------------- secrets


def test_secrets_stay_out_of_state_files_and_errors(world):
    world.vast.templates = [template_row("new", 300, CURRENT_ENV)]
    world.vast.instances = {555: inst_row(555, extra_env=[["HF_TOKEN", ENV_SECRET], ["RULZ_REPO", "r"]])}
    template = mk.default_template(world.templates())
    rental = rent_it(world, template=template)
    out = json.dumps([rental, world.account(), world.instances(), world.templates(), world.rentals(),
                      world.search(), world.prefs()], default=str)
    assert ENV_SECRET not in out and KEY not in out
    for path in world.home.rglob("*"):
        if path.is_file():
            assert KEY not in path.read_text(encoding="utf-8", errors="replace"), path
            assert ENV_SECRET not in path.read_text(encoding="utf-8", errors="replace"), path
    world.vast.rent_answer = (400, json.dumps({"msg": f"key {KEY} rejected"}))
    world._drop_intent("")
    world.vast.instances = {}
    with pytest.raises(mk.MarketError) as e:
        rent_it(world)
    assert KEY not in str(e.value) and KEY not in e.value.message


def test_remote_instances_keep_the_env_in_memory_only(world):
    world.vast.instances = {1: inst_row(1, extra_env=[["HF_TOKEN", ENV_SECRET], ["RULZ_REPO", "r"]])}
    rows = mk.remote_instances(world.api)
    assert rows[0]["extra_env"] == {"HF_TOKEN": ENV_SECRET, "RULZ_REPO": "r"}
    assert remote.is_rulztrainer(rows[0])
    assert mk.parse_instance(rows[0]).get("extra_env") is None


# ---------------------------------------------------------------- preferences and catalog


def test_prefs_are_validated_and_saved(world):
    assert world.prefs() == {"filters": mk.default_filters(), "template": None, "after": "stop",
                             "idle_minutes": 60, "bucket": ""}
    got = world.set_prefs(filters={"disk": 99999, "bogus": 1}, template="abc", after="destroy", idle_minutes=5,
                          bucket="me/bucket", nothing="dropped")
    assert got["filters"]["disk"] == 4000 and got["template"] == "abc" and got["after"] == "destroy"
    assert got["idle_minutes"] == 10 and got["bucket"] == "me/bucket" and "nothing" not in got
    assert world.prefs() == got
    with pytest.raises(mk.MarketError):
        world.set_prefs(after="explode")
    with pytest.raises(mk.MarketError):
        world.set_prefs(idle_minutes="soon")
    assert world.set_prefs(template=None)["template"] is None and world.prefs()["after"] == "destroy"
    assert world.search(None)  # the saved filters are used when none are given


def test_catalog_is_cached_a_day_and_falls_back(world):
    world.vast.offers = [offer_row(1, gpu_name="RTX 4090", geolocation="Texas, US"),
                         offer_row(2, gpu_name="H100 SXM", geolocation="Berlin, DE"),
                         offer_row(3, gpu_name=None, geolocation="Mars")]
    c = mk.catalog(world.api, world.home, world.clock.now)
    assert c["gpus"] == ["H100 SXM", "RTX 4090"] and c["live"] is True
    assert [x["code"] for x in c["countries"]] == ["DE", "US"] and c["countries"][0]["name"] == "Germany"
    n = len(world.vast.of("POST", "bundles"))
    mk.catalog(world.api, world.home, world.clock.now)
    assert len(world.vast.of("POST", "bundles")) == n  # cached
    world.clock.sleep(86401)
    mk.gpu_names(world.api, world.home, world.clock.now)
    assert len(world.vast.of("POST", "bundles")) == n + 1
    world.vast.fail = lambda m, p: OSError()
    world.clock.sleep(86401)
    c = mk.catalog(world.api, world.home, world.clock.now)  # the search fails: the last list is kept
    assert c["gpus"] == ["H100 SXM", "RTX 4090"]
    fresh = mk.catalog(world.api, world.home / "none", world.clock.now)
    assert fresh["gpus"] == mk.FALLBACK_GPUS and fresh["live"] is False and len(fresh["countries"]) > 50
    assert mk.country_list(world.api, world.home / "none", world.clock.now)[0]["code"]


# ---------------------------------------------------------------- rulz_remote.list_instances


def test_list_instances_uses_the_rest_list_when_there_is_a_key(monkeypatch, world):
    world.vast.instances = {1: inst_row(1, extra_env=[["RULZ_REPO", "r"]]), 2: inst_row(2)}
    monkeypatch.setattr(mk, "api_key", lambda: KEY)
    monkeypatch.setattr(mk, "Api", lambda *a, **k: world.api)
    monkeypatch.setattr(remote, "vastai", lambda *a: pytest.fail("the CLI was used"))
    rows = remote.list_instances()
    assert [r["id"] for r in rows] == [1, 2] and rows[0]["extra_env"] == {"RULZ_REPO": "r"}
    assert remote.is_rulztrainer(rows[0]) and not remote.is_rulztrainer(rows[1])
    assert remote.pick_instance(rows, 1)["id"] == 1


def test_list_instances_falls_back_to_the_cli(monkeypatch, world):
    cli = json.dumps([{"id": 3, "actual_status": "running"}])
    monkeypatch.setattr(remote, "vastai", lambda *a: cli)
    monkeypatch.setattr(mk, "api_key", lambda: None)  # no key
    assert remote.list_instances() == [{"id": 3, "actual_status": "running"}]
    monkeypatch.setattr(mk, "api_key", lambda: KEY)  # a key, but the list cannot be read
    monkeypatch.setattr(mk, "Api", lambda *a, **k: mk.Api(opener=lambda *a: (500, ""), key=KEY))
    assert remote.list_instances() == [{"id": 3, "actual_status": "running"}]


def test_the_market_does_not_import_the_fleet():
    source = (REPO / "scripts" / "vast" / "rulz_market.py").read_text(encoding="utf-8")
    assert "rulz_fleet" not in source.replace("rulz_fleet.py", "")


# ---------------------------------------------------------------- review fixes


def test_status_kinds_with_a_json_body_are_definite():
    for status in (401, 403, 429):
        api = mk.Api(opener=lambda *a, st=status: (st, json.dumps({"msg": "no"})), key=KEY, sleep=lambda s: None)
        with pytest.raises(mk.MarketError) as e:
            api.call("GET", "/x", retry429=False)
        assert e.value.definite is True


def test_a_malformed_or_unreadable_rentals_file_is_never_overwritten(world):
    rent_it(world)
    path = world.home / "rentals.json"
    world.vast.instances = {555: inst_row(555)}
    path.write_text("{ not json", encoding="utf-8")
    for call in (world.rentals, world.intents, lambda: rent_it(world), lambda: world.destroy(555)):
        with pytest.raises(mk.MarketError) as e:
            call()
        assert e.value.kind == "state" and "rentals.json" in e.value.message
    assert path.read_text(encoding="utf-8") == "{ not json"
    path.write_text("[1, 2]", encoding="utf-8")  # valid JSON, wrong shape
    with pytest.raises(mk.MarketError):
        world.rentals()
    path.unlink()  # only a missing file starts empty
    assert world.rentals() == [] and world.intents() == []


def test_a_sharing_violation_is_retried_then_raises(world, monkeypatch):
    rent_it(world)
    real = Path.read_text
    seen = []

    def flaky(self, *a, **k):
        if self.name == "rentals.json":
            seen.append(1)
            if len(seen) < 3:
                raise PermissionError("in use")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", flaky)
    assert len(world.rentals()) == 1 and len(seen) == 3

    def stuck(self, *a, **k):
        if self.name == "rentals.json":
            raise PermissionError("in use")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", stuck)
    with pytest.raises(mk.MarketError) as e:
        world.rentals()
    assert e.value.kind == "state"
    monkeypatch.setattr(Path, "read_text", real)
    assert len(world.rentals()) == 1  # nothing was lost


def test_a_broken_market_file_keeps_its_other_keys_on_write(world):
    world.set_prefs(bucket="me/b")
    path = world.home / "market.json"
    good = path.read_text(encoding="utf-8")
    path.write_text("garbage", encoding="utf-8")
    assert world.prefs() == mk._prefs_out({})  # a read falls back to the defaults
    with pytest.raises(mk.MarketError) as e:
        world.set_prefs(bucket="other")
    assert e.value.kind == "state" and path.read_text(encoding="utf-8") == "garbage"
    world.vast.offers = [offer_row(1)]
    mk.catalog(world.api, world.home, world.clock.now)  # a cache that cannot be written is skipped quietly
    assert path.read_text(encoding="utf-8") == "garbage"
    path.write_text(good, encoding="utf-8")
    assert world.prefs()["bucket"] == "me/b"


def test_a_sending_intent_of_a_live_process_is_not_settled_as_crashed(world):
    world.vast.home = world.home
    other = {}

    def during_the_put(body):
        # a second window of the app, while the first is inside rent()
        again = mk.Market(home=world.home, api=world.api, now=world.clock.now, sleep=world.clock.sleep,
                          which=lambda n: "ssh", ssh_dir=world.ssh_path)
        other["settle"] = again.settle()
        other["intent"] = again.intents()[0]["state"]
        with pytest.raises(mk.MarketError) as e:
            again.rent(an_offer(i=100), mk.default_filters())
        other["kind"] = e.value.kind
        with pytest.raises(mk.MarketError) as e2:
            again.check_intent(again.intents()[0]["label"])
        other["check"] = e2.value.kind

    world.vast.rent_effect = during_the_put
    rent_it(world)
    assert other == {"settle": [], "intent": "sending", "kind": "busy", "check": "busy"}
    # the lock is free again once the rent returned, so a leftover sending intent is a crash
    (world.home / "rentals.json").write_text(json.dumps({"rentals": [], "intents": [
        {"label": "rulz-trainer-deadbeef", "offer": 1, "price": 1.0, "created": world.clock.now(), "disk": 1,
         "state": "sending"}]}), encoding="utf-8")
    world.settle()
    assert world.intents()[0]["state"] == "checking"


def test_rent_refuses_when_the_rent_lock_is_held(world):
    held = mk._try_lock(world.home / "rent.lock")
    assert held is not None
    try:
        with pytest.raises(mk.MarketError) as e:
            rent_it(world)
        assert e.value.kind == "busy" and e.value.definite and world.vast.of("PUT", "asks") == []
    finally:
        held.close()
    assert rent_it(world)["id"] == 555


def test_instance_lists_are_not_trusted_when_a_row_or_the_count_is_off():
    def lister(page):
        return mk.Api(opener=lambda *a: (200, json.dumps(page)), key=KEY)

    with pytest.raises(mk.MarketError) as e:
        lister({"instances": [inst_row(1), {"label": "no id"}]}).instances_raw()
    assert e.value.kind == "unreadable"
    for bad in ("x", None, 0, -4):
        with pytest.raises(mk.MarketError):
            lister({"instances": [inst_row(1), {"id": bad}]}).instances_raw()
    with pytest.raises(mk.MarketError):
        lister({"instances": ["junk"]}).instances_raw()
    with pytest.raises(mk.MarketError) as e:
        lister({"instances": [inst_row(1)], "total_instances": 2}).instances_raw()
    assert "count" in e.value.message
    assert len(lister({"instances": [inst_row(1)], "total_instances": 1}).instances_raw()) == 1
    assert len(lister({"instances": [inst_row(1)]}).instances_raw()) == 1


def test_scrub_text():
    out = mk.scrub_text("mail me@example.com see https://x.io/a?token=abc and HF_TOKEN=hf_secretvalue "
                        "also abcdefghijklmnopqrstuvwxyz0123 ok")
    assert "me@example.com" not in out and "token=abc" not in out and "hf_secretvalue" not in out
    assert "abcdefghijklmnopqrstuvwxyz0123" not in out and out.endswith("ok")
    assert "HF_TOKEN=[hidden]" in out
    assert mk.scrub_text("x" * 5 + " " + "y" * 500, 50) == ("x" * 5 + " [hidden]")[:50]
    assert len(mk.scrub_text("a b " * 200, 30)) <= 30 and mk.scrub_text(None) == ""
    assert mk.scrub_text("Container failed to start") == "Container failed to start"


def test_status_msg_and_error_texts_are_scrubbed(world):
    msg = "Error: image pull failed for user@host.com, key=SUPERSECRETVALUE url https://h/x?sig=abc " + "z" * 600
    row = mk.parse_instance(inst_row(1, status_msg=msg))
    for bad in ("user@host.com", "SUPERSECRETVALUE", "sig=abc"):
        assert bad not in row["status_msg"]
    assert len(row["status_msg"]) <= 400
    api = mk.Api(opener=lambda *a: (400, json.dumps({"msg": "bad key=SECRETSECRETSECRET for a@b.co"})), key=KEY)
    with pytest.raises(mk.MarketError) as e:
        api.get("/x")
    assert "SECRETSECRET" not in e.value.message and "a@b.co" not in e.value.message


@pytest.mark.parametrize("answer", [(200, json.dumps({})), (200, json.dumps({"msg": "ok"})), (200, "ok"),
                                    (200, "[]"), (200, json.dumps({"success": "true"}))])
def test_start_stop_bid_destroy_need_success_true(world, answer):
    world.vast.instances = {5: inst_row(5, actual_status="exited", intended_status="stopped")}
    world.vast.put_answer = answer
    for call in (lambda: world.start(5), lambda: world.stop(5), lambda: world.change_bid(5, 0.5)):
        with pytest.raises(mk.MarketError) as e:
            call()
        assert e.value.kind == "unclear" and e.value.definite is False and "vast.ai" in e.value.message
    world.vast.put_answer = (200, json.dumps({"success": True}))
    assert world.start(5)["started"] and world.stop(5)["ok"] and world.change_bid(5, 0.5)["price"] == 0.5


def test_destroy_unclear_rereads_gone_counts_as_destroyed(world):
    world.vast.instances = {9: inst_row(9)}
    calls = []
    orig = world.api.call

    def unclear(method, path, *a, **k):
        if method == "DELETE":
            calls.append(path)
            world.vast.instances.pop(9, None) if world.vast.gone_on_delete else None
            return mk.Reply(200, {"msg": "hm"}, True)
        return orig(method, path, *a, **k)

    world.api.call = unclear
    world.vast.gone_on_delete = True
    assert world.destroy(9) == {"id": 9, "ok": True}
    world.vast.instances = {9: inst_row(9)}
    world.vast.gone_on_delete = False
    with pytest.raises(mk.MarketError) as e:  # still there: not destroyed, said so
        world.destroy(9)
    assert e.value.kind == "unclear" and len(calls) == 2


def test_destroy_network_failure_with_instance_gone_counts_and_a_refusal_does_not(world):
    world.vast.instances = {9: inst_row(9)}
    world.vast.home = None
    gone = {"after": False}

    def fail(method, path):
        if method == "DELETE":
            world.vast.instances.pop(9, None)
            return OSError("dropped")
        return None

    world.vast.fail = fail
    assert world.destroy(9)["ok"] is True
    world.vast.fail = None
    world.vast.instances = {9: inst_row(9)}
    world.vast.put_answer = (200, json.dumps({"success": False, "msg": "no"}))
    world.api.call = lambda m, p, *a, **k: mk.Reply(200, {"success": False, "msg": "no"}, True) if m == "DELETE" \
        else mk.Api.call(world.api, m, p, *a, **k)
    with pytest.raises(mk.MarketError) as e:
        world.destroy(9)
    assert e.value.kind == "refused"
    assert gone == {"after": False}


def test_interruptible_rent_refuses_when_min_bid_or_storage_is_missing(world):
    f = dict(mk.default_filters(), type="bid")
    for missing in ("min_bid", "storage_total_cost"):
        world.vast.offers = [offer_row(100, dph_base=0.2, dph_total=0.28, **{missing: None})]
        shown = mk.parse_offer(offer_row(100, dph_base=0.2, min_bid=0.2, dph_total=0.28), "bid")
        with pytest.raises(mk.MarketError) as e:
            world.rent(shown, f, bid=0.5)
        assert e.value.kind == "price" and e.value.definite
    assert world.vast.of("PUT", "asks") == [] and world.intents() == []


def test_on_demand_rent_needs_the_fresh_price(world):
    world.vast.offers = [offer_row(100, dph_total=None)]
    with pytest.raises(mk.MarketError) as e:
        rent_it(world)
    assert e.value.kind == "gone" and world.vast.of("PUT", "asks") == []


def test_stop_state(world):
    def look(**kw):
        world.vast.instances = {5: inst_row(5, **kw)}
        return world.stop_state(5)

    assert look(actual_status="exited", intended_status="stopped") == {"stopped": True, "status": "exited",
                                                                       "known": True}
    for status in ("stopped", "offline", "inactive"):
        assert look(actual_status=status, intended_status="stopped")["stopped"] is True
    assert look(actual_status="running", intended_status="stopped") == {"stopped": False, "status": "running",
                                                                        "known": True}
    assert look(actual_status="exited", intended_status="running")["stopped"] is False  # on its way back up
    assert look(actual_status="loading", intended_status="stopped")["stopped"] is False
    unknown = look(actual_status=None, intended_status="stopped")
    assert unknown == {"stopped": False, "status": "stopped", "known": False}
    world.vast.instances = {}
    assert world.stop_state(5) == {"stopped": True, "status": "gone", "known": True}
    world.vast.instances = {5: inst_row(5)}
    world.vast.fail = lambda m, p: OSError()
    assert world.stop_state(5) == {"stopped": False, "status": "unknown", "known": False}


def test_watch_stop_does_not_take_a_missing_status_for_stopped(world):
    world.vast.instances = {5: inst_row(5, actual_status=None, intended_status="stopped")}
    out = world.watch_stop(5)
    assert out["stopped"] is False and "alert" in out and world.clock.now() - 1_000_000.0 == 600
    world.vast.instances = {5: inst_row(5, actual_status="exited", intended_status="running")}
    assert world.watch_stop(5)["stopped"] is False


def test_the_cli_answer_must_be_a_list_for_list_instances(monkeypatch):
    monkeypatch.setattr(mk, "api_key", lambda: None)
    for answer in ('"text"', "42", "null", '{"other": 1}', '{"instances": "x"}', "not json"):
        monkeypatch.setattr(remote, "vastai", lambda *a, ans=answer: ans)
        with pytest.raises(remote.Unreachable):
            remote.list_instances()
    monkeypatch.setattr(remote, "vastai", lambda *a: json.dumps({"instances": [{"id": 1}]}))
    assert remote.list_instances() == [{"id": 1}]
    monkeypatch.setattr(remote, "vastai", lambda *a: json.dumps([{"id": 2}]))
    assert remote.list_instances() == [{"id": 2}]


def test_a_partial_rest_list_is_never_returned(monkeypatch):
    cli = json.dumps([{"id": 3}])
    monkeypatch.setattr(remote, "vastai", lambda *a: cli)
    monkeypatch.setattr(mk, "api_key", lambda: KEY)
    page = {"instances": [inst_row(1)], "next_token": "t"}
    broken = mk.Api(opener=lambda *a: (200, json.dumps(page)) if "after_token" not in a[1] else (500, ""), key=KEY)
    monkeypatch.setattr(mk, "Api", lambda *a, **k: broken)
    assert remote.list_instances() == [{"id": 3}]  # the second page failed: the CLI answers, not page one alone
