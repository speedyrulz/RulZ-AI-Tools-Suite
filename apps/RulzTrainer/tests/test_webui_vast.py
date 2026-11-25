"""The WebUI's vast.ai parts (rulztrainer/vastui.py and its hooks in webui.py): the credit chip, the filters' counts,
what a row can do, the send dialog's readiness, the account watch and the sends loop, with fakes. Nothing here touches
the network, spends money or uses a GPU; the drawing itself is checked in a browser by the previews."""
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "vast"))
pytest.importorskip("nicegui")
pytest.importorskip("psutil")

import rulz_market as mk  # noqa: E402
from rulztrainer import vastui as v  # noqa: E402

NOW = 1_000_000.0


def snap(credit=11.44, run=0.53, rest=0.04, gpus=1, hours=None, alerts=(), error="", key=True):
    per = run + rest
    acct = {"credit": credit, "instances": [],
            "burn": {"per_hour": per, "running": 1 if run else 0, "stopped": 1 if rest else 0, "gpus": gpus,
                     "disks": 400, "running_per_hour": run, "stopped_per_hour": rest},
            "hours_left": hours if hours is not None else (credit / per if per else None),
            "empty_at": NOW + 3600 * (credit / per) if per else None}
    return {"account": acct, "error": error, "at": NOW, "key": key, "alerts": list(alerts), "stopping": {}}


# ---------------------------------------------------------------- formatting


def test_prices_have_two_decimals_and_a_third_only_when_there_is_one():
    assert v.price(0.53) == "$0.53" and v.price(0.534) == "$0.534" and v.price(11.44) == "$11.44"
    assert v.price(1.2) == "$1.20" and v.price(None) == "-" and v.price("x") == "-" and v.price(True) == "-"
    assert v.per_hour(0.53) == "$0.53/hour"


def test_hours_read_as_minutes_hours_or_days():
    assert v.hours_text(0.5) == "30 min" and v.hours_text(0.001) == "1 min"
    assert v.hours_text(14.2) == "14 h" and v.hours_text(99) == "99 h" and v.hours_text(240) == "10 d"
    assert v.hours_text(None) == ""


def test_the_chip_says_credit_and_time_left_and_turns_amber_then_red():
    view = v.chip_view(snap(credit=11.44, run=0.53, rest=0.04))
    assert view["text"] == "$11.44 · 20 h left" and view["level"] == "ok" and view["billed"]
    amber = v.chip_view(snap(credit=2.0, run=0.53, rest=0.04))  # 3.5 h
    assert amber["level"] == "warning" and amber["icon"] == "warning"
    red = v.chip_view(snap(credit=0.3, run=0.53, rest=0.04))  # 31 min
    assert red["level"] == "critical" and red["text"] == "$0.30 · 32 min left"
    assert v.chip_view(snap(credit=3.6, run=0.53, rest=0.04))["level"] == "ok"  # 6.3 h
    assert v.level_of(6.0) == "ok" and v.level_of(5.99) == "warning" and v.level_of(1.0) == "warning"
    assert v.level_of(0.99) == "critical" and v.level_of(None) == "ok"


def test_the_chip_tooltip_splits_the_cost_and_says_traffic_is_not_in_it():
    tip = " | ".join(v.chip_view(snap())["tip"])
    assert "$0.53/hour for 1 running instance (1 GPU)" in tip
    assert "$0.04/hour for the disk of 1 stopped instance" in tip
    assert "runs out about" in tip and "Traffic" in tip and "not included" in tip


def test_nothing_billed_shows_credit_alone_and_only_on_the_tab():
    view = v.chip_view(snap(credit=5.0, run=0, rest=0))
    assert view["billed"] is False and view["text"] == "$5.00" and view["level"] == "ok"
    assert "Nothing is billed" in " ".join(view["tip"])


def test_a_stop_alert_makes_the_chip_red_and_names_it():
    alerts = [{"id": 7, "text": "Instance 7 was told to stop but vast.ai still reports it running."}]
    view = v.chip_view(snap(alerts=alerts))
    assert view["level"] == "critical" and view["icon"] == "error" and view["tip"][0].startswith("Instance 7")


def test_an_unreadable_account_is_a_sentence_not_a_crash():
    view = v.chip_view({"account": None, "error": "vast.ai did not answer.", "alerts": []})
    assert view["tip"] == ["vast.ai did not answer."] and "not reachable" in view["text"]
    assert v.credit_lines({"account": None, "error": "vast.ai did not answer."}) == ["vast.ai did not answer."]


def test_the_credit_strip_says_what_bills_and_how_long_it_lasts():
    lines = v.credit_lines(snap())
    assert lines[0].startswith("Billing $0.57/hour now: $0.53/hour for 1 running instance and $0.04/hour for the disks")
    assert "lasts about 20 h" in lines[1] and "Traffic is not included" in lines[1]
    assert v.credit_lines(snap(run=0, rest=0))[0].startswith("Nothing is billed now")


def test_credit_lasts_is_none_when_nothing_bills():
    assert v.lasts(10, 0) is None and v.lasts(None, 1) is None and v.lasts(10, 2) == 5.0


# ---------------------------------------------------------------- filters and offers


def test_filter_counts_count_only_what_limits_the_search():
    defaults = mk.default_filters()
    counts = v.filter_counts(defaults)
    assert set(counts) == set(mk.GROUPS)
    assert counts["Rental, GPU and region"] == 1  # one GPU; type and sort never count, any region limits nothing
    assert counts["Disk"] == 1 and counts["Reliability"] == 1 and counts["Price"] == 0
    changed = dict(defaults, secure_cloud=True, static_ip=True, dph=[None, 1.5], gpu_name=["RTX 4090"],
                   num_gpus="any", region="eu")
    after = v.filter_counts(changed)
    assert after["Machine options"] == 2 and after["Price"] == 1
    assert after["Rental, GPU and region"] == 2  # GPU model and the region; "any" GPUs no longer counts
    assert v.group_label("Price", 2) == "Price · 2 set" and v.group_label("Price", 0) == "Price"


def test_every_filter_kind_has_a_set_rule():
    kinds = {f.kind for f in mk.FILTERS}
    assert kinds <= {"range", "min", "switch", "choice", "multi", "text"}
    by = {f.key: f for f in mk.FILTERS}
    assert v.is_set(by["dph"], [None, 2]) and not v.is_set(by["dph"], [None, None])
    assert v.is_set(by["static_ip"], True) and not v.is_set(by["static_ip"], False)
    assert v.is_set(by["machine_id"], "12") and not v.is_set(by["machine_id"], "  ")
    assert not v.is_set(by["type"], "bid") and not v.is_set(by["sort"], "price_up")


def test_reserved_and_unavailable_rows_are_not_rentable():
    assert v.can_rent({"type": "ask", "rentable": True}) == (True, "")
    ok, why = v.can_rent({"type": "reserved"})
    assert not ok and "vast.ai" in why
    ok, why = v.can_rent(mk.Offer(id=1, rentable=False))
    assert not ok and why == "Not available now"


def test_an_offer_row_carries_every_column_as_text():
    o = mk.Offer(id=9, machine_id=42, gpu_name="RTX 4090", num_gpus=2, gpu_total_ram_mb=49128, dph_total=1.07,
                 dlperf=70.1, dlperf_per_dollar=65.5, tflops=165.0, reliability=0.987, inet_up=900, inet_down=1200,
                 inet_up_per_gb=0.004, inet_down_per_gb=0.006, cpu_name="EPYC", cpu_cores=24, cpu_ram_mb=128000,
                 disk_bw=2400, disk_space=900, cuda=12.8, geolocation="Texas, US", verified=True, ports=40,
                 duration_days=30)
    row = v.offer_row(o)
    assert row["gpu"] == "2x RTX 4090" and row["vram"] == "48 GB" and row["price"] == "$1.07"
    assert row["up"] == "$4/TB" and row["down"] == "$6/TB" and row["reliability"] == "98.7%"
    assert row["duration"] == "30 d" and row["net"] == "900 / 1,200 Mbps" and row["where"] == "Verified"
    assert row["machine"] == 42 and row["can_rent"] and row["cpu"] == "EPYC 24 cores"
    bare = v.offer_row(mk.Offer(id=1, type="reserved"))
    assert bare["can_rent"] is False and bare["tflops"] == "-" and bare["reliability"] == "-"


def test_an_interruptible_price_is_the_bid_plus_the_disk():
    bid = mk.Offer(id=1, type="bid", dph_total=0.3, storage_dph=0.08, min_bid=0.2)
    assert v.offer_price(bid, 0.25) == pytest.approx(0.33) and v.offer_price(bid, None) is None
    assert v.offer_price(mk.Offer(id=1, dph_total=0.53)) == 0.53


def test_a_changed_price_comes_back_on_a_copy_of_the_offer():
    ask = mk.Offer(id=1, dph_total=0.5)
    new = v.with_price(ask, 0.6)
    assert new.dph_total == 0.6 and ask.dph_total == 0.5
    bid = mk.Offer(id=1, type="bid", dph_total=0.3, storage_dph=0.08)
    assert v.with_price(bid, 0.35, bid=0.25).storage_dph == pytest.approx(0.10)
    assert v.with_price({"dph_total": 1}, 2)["dph_total"] == 2


def test_the_rent_dialog_warns_about_a_small_disk_a_weak_host_a_short_rental_and_a_bid():
    ok = mk.Offer(id=1, reliability=0.99, duration_days=30)
    assert v.rent_warnings(ok, 200) == []
    small = v.rent_warnings(ok, 80, upload_bytes=30e9)  # 60 + 30 + 20 = 110 GB needed
    assert len(small) == 1 and "110 GB" in small[0]
    weak = mk.Offer(id=1, reliability=0.9, duration_days=1.5)
    text = " ".join(v.rent_warnings(weak, 200))
    assert "90.0%" in text and "1.5 days" in text
    bid = mk.Offer(id=1, type="bid", min_bid=0.2, reliability=0.99, duration_days=30)
    out = v.rent_warnings(bid, 200, bid=0.21)
    assert any("pauses this instance" in w for w in out) and any("close to the minimum" in w for w in out)
    assert any("under the minimum" in w for w in v.rent_warnings(bid, 200, bid=0.1))


def test_upload_time_says_what_it_assumes():
    assert v.upload_seconds(12.5e6, 100) == pytest.approx(1.0) and v.upload_seconds(1, 0) is None
    text = v.upload_text(11.9 * 1024 ** 3, 100)
    assert "11.9 GB" in text and "100 Mbit/s" in text and "about" in text


def test_the_cost_block_shows_the_price_the_credit_and_how_long_it_lasts():
    rows = dict(v.cost_lines(11.44, 0.57, 0.53))
    assert rows["Instance"] == "$0.53/hour" and rows["Credit"] == "$11.44"
    assert rows["Credit lasts"] == "about 10 h with this instance (now 20 h)"
    assert "Estimated run cost" not in rows
    run = dict(v.cost_lines(11.44, 0.57, 0.53, run_hours=4))
    assert run["Estimated run cost"] == "about $2.12 for 4 h of GPU time"
    assert dict(v.cost_lines(11.44, 0.57, 0))["Instance"] == "already billing"


# ---------------------------------------------------------------- the send dialog


def plan(files=(), blockers=(), datasets=()):
    return SimpleNamespace(files=list(files), blockers=list(blockers), datasets=list(datasets), changes=[])


FILE_A = {"keys": ["transformer_file"], "path": "D:/m/custom.safetensors", "bytes": 12_000_000_000, "kind": "file",
          "standard": None}
FILE_B = {"keys": ["vae_file"], "path": "D:/m/vae.safetensors", "bytes": 335_000_000, "kind": "file",
          "standard": {"repo": "x/y", "file": "ae.safetensors", "bytes": None}}
INSTANCE = {"kind": "instance", "row": {"id": 1, "status": "running", "ours": True}}


def test_a_file_with_a_standard_file_defaults_to_it_and_one_without_must_be_chosen():
    assert v.default_choice(FILE_B) == "standard" and v.default_choice(FILE_A) is None
    assert [k for k, _ in v.file_options(FILE_A)] == ["upload", "bucket"]
    assert [k for k, _ in v.file_options(FILE_B)] == ["upload", "bucket", "standard"]
    assert v.standard_text(FILE_B) == "ae.safetensors from x/y" and v.standard_text(FILE_A) == ""


def test_the_red_button_waits_for_a_choice_for_every_local_file():
    p = plan([FILE_A, FILE_B])
    ready, why = v.send_state(p, {FILE_A["path"]: None, FILE_B["path"]: "standard"}, target=INSTANCE)
    assert not ready and why == ["Choose how to send custom.safetensors."]
    ready, why = v.send_state(p, {FILE_A["path"]: "upload", FILE_B["path"]: "standard"}, target=INSTANCE)
    assert ready and why == []
    # "standard" is not a choice for a file that has none
    ready, why = v.send_state(p, {FILE_A["path"]: "standard", FILE_B["path"]: "standard"}, target=INSTANCE)
    assert not ready


def test_a_bucket_route_needs_a_confirmed_bucket():
    p = plan([FILE_A])
    picks = {FILE_A["path"]: "bucket"}
    assert v.needs_bucket(picks) and v.needs_bucket({"p": {"how": "bucket"}}) and not v.needs_bucket({"p": "upload"})
    ready, why = v.send_state(p, picks, bucket="me/b", bucket_ok=False, target=INSTANCE)
    assert not ready and why == ["Confirm the Hugging Face bucket."]
    ready, _ = v.send_state(p, picks, bucket="", bucket_ok=True, target=INSTANCE)
    assert not ready
    ready, _ = v.send_state(p, picks, bucket="me/b", bucket_ok=True, target=INSTANCE)
    assert ready
    # a plan that uses no bucket never asks for one
    assert v.send_state(p, {FILE_A["path"]: "upload"}, target=INSTANCE)[0]


def test_blockers_a_missing_target_and_a_busy_send_all_keep_the_button_off():
    blockers = [{"key": "dataset_dir", "path": "D:/gone", "reason": "missing dataset folder"}]
    ready, why = v.send_state(plan(blockers=blockers), {}, target=INSTANCE)
    assert not ready and why == ["missing dataset folder: D:/gone"]
    ready, why = v.send_state(plan(), {}, target=None)
    assert not ready and why == ["Pick an instance, or rent a new one."]
    ready, why = v.send_state(plan(), {}, target=INSTANCE, busy=True)
    assert not ready
    assert v.send_state(None, {}, target=INSTANCE)[0] is False
    assert v.send_state(plan(), {}, target=INSTANCE)[0] is True  # nothing to upload, an instance: ready


def test_destroy_is_refused_for_an_instance_this_app_did_not_rent():
    ready, why = v.send_state(plan(), {}, target=INSTANCE, after="destroy", destroy_allowed=False)
    assert not ready and "Destroy is only for instances this app rented" in why[0]


def test_choices_go_to_the_manager_with_the_bucket_only_where_it_is_used():
    p = plan([FILE_A, FILE_B])
    out = v.choices_for(p, {FILE_A["path"]: "bucket", FILE_B["path"]: "standard"}, "me/b")
    assert out == {FILE_A["path"]: {"how": "bucket", "bucket": "me/b"}, FILE_B["path"]: {"how": "standard"}}
    assert v.choices_for(p, {FILE_A["path"]: None}, "me/b") == {}


def test_after_the_run_stop_is_the_default_and_destroy_is_only_for_our_own_rentals():
    assert [k for k, _ in v.after_options(ours=True)] == ["stop", "destroy", "leave"]
    assert [k for k, _ in v.after_options(ours=False)] == ["stop", "leave"]
    assert v.after_default("stop", ours=True) == "stop" and v.after_default("destroy", ours=True) == "destroy"
    assert v.after_default("destroy", ours=False) == "leave" and v.after_default("junk", ours=True) == "stop"
    assert "disk stays and costs $0.02/hour" in v.after_text("stop", 0.02)
    assert "for good" in v.after_text("destroy") and "keeps billing $0.53/hour" in v.after_text("leave", None, 0.53)


def test_an_instance_row_says_price_what_runs_free_disk_and_version():
    row = {"gpu_name": "RTX 4090", "num_gpus": 1, "per_hour": 0.53, "job": {"job": "a"}, "free_gb": 142.4,
           "version": "4.4.0"}
    assert v.instance_row_text(row) == ["1x RTX 4090", "$0.53/hour", "running a", "142 GB free", "RulZ AI Trainer 4.4.0"]
    idle = v.instance_row_text({"gpu_name": "RTX 3090", "num_gpus": 2, "dph_total": 0.6})
    assert idle == ["2x RTX 3090", "$0.60/hour", "idle"]
    assert v.instance_state_word({"status": "running"})[1] == "Running"
    assert v.instance_state_word({"status": "exited"})[1] == "Exited"


# ---------------------------------------------------------------- a send's steps


def test_cancel_is_offered_only_until_the_run_is_queued():
    steps = lambda **st: [{"key": k, "state": s} for k, s in st.items()]  # noqa: E731
    assert v.send_can_cancel({"state": "going", "steps": steps(rent="done", files="going", queue="waiting")})
    assert not v.send_can_cancel({"state": "going", "steps": steps(files="done", queue="going")})
    assert not v.send_can_cancel({"state": "running", "steps": steps(queue="done", run="going")})
    assert not v.send_can_cancel({"state": "done", "steps": steps(rent="done")})


def test_a_steps_line_adds_the_percentage_while_it_goes():
    assert v.step_line({"state": "going", "text": "photos: 31 of 62 files", "frac": 0.5}) == "photos: 31 of 62 files 50%"
    assert v.step_line({"state": "going", "text": "7 of 12 GB, 50%", "frac": 0.5}) == "7 of 12 GB, 50%"
    assert v.step_line({"state": "done", "text": "", "frac": 1.0}) == ""
    assert v.countdown_text(154) == "2:34" and v.countdown_text(None) == "" and v.countdown_text(-5) == "0:00"
    assert v.send_signature({"steps": [{"key": "rent"}, {"key": "boot"}]}) == "rent,boot"
    assert set(v.STEP_LOOK) == {"waiting", "going", "done", "failed", "skipped"}
    assert set(v.STATE_LOOK) >= {"going", "waiting", "running", "done", "failed", "cancelled", "interrupted"}


def test_every_status_look_is_an_icon_a_word_and_a_level():
    for table in (v.STEP_LOOK, v.STATE_LOOK):
        for icon, word, level in table.values():
            assert icon and word and level in ("ok", "warn", "err", "muted")


# ---------------------------------------------------------------- the account watch and the sends loop


class FakeServices:
    def __init__(self, market=None, key=True, manager=None):
        self.market, self.key, self.manager = market, key, manager

    def get_market(self):
        return self.market

    def get_manager(self):
        return self.manager

    def has_key(self):
        return self.key


class FakeMarket:
    def __init__(self, account=None, error=None, stop=None):
        self.acct, self.error, self.stop, self.calls = account, error, stop, []

    def account(self):
        self.calls.append("account")
        if self.error:
            raise self.error
        return self.acct

    def watch_stop(self, instance_id):
        self.calls.append(("watch_stop", instance_id))
        return self.stop


def test_the_watch_reads_the_account_and_keeps_the_last_value_on_an_error():
    market = FakeMarket(snap()["account"])
    watch = v.AccountWatch(FakeServices(market), now=lambda: NOW)
    first = watch.poll()
    assert first["key"] is True and first["account"]["credit"] == 11.44 and first["error"] == ""
    market.error = mk.MarketError("vast.ai is busy. Try again.", "rate")
    again = watch.poll()
    assert again["error"] == "vast.ai is busy. Try again." and again["account"]["credit"] == 11.44


def test_an_error_that_is_not_ours_is_a_plain_sentence():
    market = FakeMarket(error=RuntimeError("Traceback (most recent call last): secret path C:/x"))
    watch = v.AccountWatch(FakeServices(market), now=lambda: NOW)
    snapshot = watch.poll()
    assert snapshot["error"] == "vast.ai did not answer." and snapshot["account"] is None


def test_without_a_key_the_watch_never_calls_vast_ai():
    market = FakeMarket(snap()["account"])
    watch = v.AccountWatch(FakeServices(market, key=False), now=lambda: NOW)
    snapshot = watch.poll()
    assert snapshot["key"] is False and snapshot["account"] is None and market.calls == []
    assert v.AccountWatch(FakeServices(None)).poll()["key"] is False


def test_a_stop_that_vast_ai_still_shows_running_becomes_a_loud_alert():
    alert = "Instance 7 was told to stop but vast.ai still reports it running after 10 minutes."
    market = FakeMarket(stop={"stopped": False, "alert": alert})
    watch = v.AccountWatch(FakeServices(market), now=lambda: NOW)
    watch.follow_stop(market, 7)
    snapshot = watch.snapshot()
    assert snapshot["alerts"][0]["text"] == alert and snapshot["stopping"] == {}
    assert v.chip_view({**snap(), "alerts": snapshot["alerts"]})["level"] == "critical"
    watch.dismiss_alert(7)
    assert watch.snapshot()["alerts"] == []
    ok = FakeMarket(stop={"stopped": True, "status": "stopped", "waited": 3})
    watch.follow_stop(ok, 8)
    assert watch.snapshot()["alerts"] == []


def test_a_failing_stop_watch_still_ends_in_an_alert_without_a_traceback():
    class Boom(FakeMarket):
        def watch_stop(self, instance_id):
            raise RuntimeError("Traceback (most recent call last)")

    watch = v.AccountWatch(FakeServices(), now=lambda: NOW)
    watch.follow_stop(Boom(), 9)
    text = watch.snapshot()["alerts"][0]["text"]
    assert "instance 9" in text and "Traceback" not in text and "still bills" in text


def test_the_sends_loop_holds_steps_and_survives_a_bad_beat():
    calls = []

    class Manager:
        def hold(self):
            calls.append("hold")
            return True

        def step(self):
            calls.append("step")
            if calls.count("step") == 1:
                raise RuntimeError("one bad beat")

    sleeps = []
    v.sends_loop(FakeServices(manager=Manager()), every=3.0, sleep=sleeps.append, rounds=3)
    assert calls.count("step") == 3 and calls.count("hold") == 3 and sleeps == [3.0, 3.0, 3.0]


def test_a_missing_send_manager_only_slows_the_beat():
    sleeps = []
    v.sends_loop(FakeServices(manager=None), every=3.0, sleep=sleeps.append, rounds=2)
    assert sleeps == [30.0, 30.0]


def test_a_module_that_cannot_be_loaded_is_a_sentence_and_no_crash():
    services = v.Services()
    assert services.module("rulz_no_such_module") is None
    assert "could not be loaded" in services.errors["rulz_no_such_module"]
    services.install(market=object(), has_key=False)
    assert services.has_key() is False and services.available() is True


def test_an_error_is_shown_as_its_own_message_or_the_fallback():
    assert v.sentence(mk.MarketError("The price rose.", "price")) == "The price rose."
    assert v.sentence(RuntimeError("Traceback...")) == "That did not work."
    assert v.sentence(OSError("disk full")) == "disk full"
    assert v.sentence(RuntimeError("x"), "Fallback.") == "Fallback."


# ---------------------------------------------------------------- the hooks in webui.py


import webui  # noqa: E402

SRC = Path(v.__file__).read_text(encoding="utf-8")


def test_vastui_never_imports_webui():
    assert not re.search(r"^\s*(import|from)\s+webui\b", SRC, re.M)


def test_every_rz_vast_class_vastui_uses_is_styled():
    used = set(re.findall(r"rz-vast-[a-z-]+", SRC))
    missing = sorted(c for c in used if f".{c}" not in webui.THEME_CSS)
    assert not missing, missing


def test_the_new_styles_use_the_themes_variables_not_colours_of_their_own():
    block = webui.THEME_CSS.split("/* Rent GPUs and sending to vast.ai")[1].split(".rz-vast-wide")[0]
    assert "#" not in block.replace("&#", "")  # no hex colours: the dark and light looks come from --rz-*


def test_the_page_has_the_tab_the_chip_and_the_run_on_control():
    page = Path(webui.__file__).read_text(encoding="utf-8")
    assert 'tab_names = ("Wizard", "Train", "Dataset", "Monitor", "Samples", "Training maps", "Previous Runs", LT_TAB, ' \
           '"Fleet",\n                 "Rent GPUs", "Help")' in page
    assert v.TAB == "Rent GPUs" and v.RUN_ON_KEY == "rulz.runon"
    assert '("Train", "Dataset", "Monitor", tab_train, tab_dataset,' in page  # the action bar is on Dataset too
    assert "vast_ui.build_run_on()" in page and "vast_ui.build_chip()" in page
    assert "vastui.start_loops" in page and "build_fleet_tab(form_config, charts, vast_ui.build_sends_card" in page


def test_nothing_of_it_shows_on_a_vast_instance():
    page = Path(webui.__file__).read_text(encoding="utf-8")
    assert "if VAST_INSTANCE is None and vastui is not None:\n        vast_ui = vastui.VastUI(" in page
    assert "if VAST_INSTANCE is None:  # this computer looks after the vast.ai instances" in page


def test_the_fleet_tab_no_longer_claims_nothing_rents():
    page = Path(webui.__file__).read_text(encoding="utf-8")
    assert "Nothing here rents, stops or destroys" not in page and "Nothing here rents or destroys" not in page
    assert "Training on vast.ai" in webui.HELP and "Run on" in webui.HELP and "Send to vast.ai" in webui.HELP


def test_a_run_going_here_or_a_job_name_taken_here_does_not_block_a_send(monkeypatch):
    from rulztrainer.config import TrainConfig

    monkeypatch.setattr(webui, "_running", lambda: True)
    monkeypatch.setattr(webui, "job_name_refusal", lambda cfg, busy=frozenset(): "Job name 'a' is taken here.")
    cfg = TrainConfig()
    local = [t for k, t in webui.review_issues(cfg, None, None, 1) if k == "error"]
    assert any("already running" in t for t in local) and any("taken here" in t for t in local)
    remote = [t for k, t in webui.review_issues(cfg, None, None, 1, remote=True) if k == "error"]
    assert not any("already running" in t for t in remote) and not any("taken here" in t for t in remote)


def test_the_wizard_follows_run_on():
    wiz = webui.Wizard.__new__(webui.Wizard)
    wiz.vast = None
    assert wiz.is_remote() is False
    wiz.vast = SimpleNamespace(remote=lambda: True)
    assert wiz.is_remote() is True
    wiz.vast = SimpleNamespace(remote=lambda: False)
    assert wiz.is_remote() is False


def test_pages_share_one_run_on_value_and_follow_each_other():
    ui_ = SimpleNamespace(run_on="pc", controls=[], listeners=[])
    seen = []
    # the value logic without a browser: set_run_on only reads and writes these attributes
    def set_run_on(self, value, save=False):
        if value not in ("pc", "vast") or value == self.run_on:
            return
        self.run_on = value
        for fn in list(self.listeners):
            fn(value)

    ui_.listeners.append(seen.append)
    set_run_on(ui_, "vast")
    set_run_on(ui_, "vast")  # unchanged: no second call
    set_run_on(ui_, "junk")
    assert ui_.run_on == "vast" and seen == ["vast"]


def test_a_search_is_only_ever_a_search():
    """Only the red buttons of the rent and send dialogs reach Market.rent / start / create."""
    calls = [m.group(0) for m in re.finditer(r"market\.(rent|start|stop|destroy|change_bid)\b|manager\.create\b", SRC)]
    assert calls  # they are there ...
    # ... and none of them is in the market view's search, sort, reset or Enter handlers
    view = SRC.split("class MarketView")[1].split("# -- renting")[0]
    assert not re.search(r"(market|manager)\.(rent|start|stop|destroy|create|change_bid)\(", view)
    assert 'keydown.enter", enter' in view and "enter = self.search" in view


def test_time_is_formatted_in_local_time():
    t = time.mktime((2026, 11, 25, 3, 12, 0, 0, 0, -1))
    assert v.when_text(t) == "Wed 25 Nov, 03:12"
    assert v.size_text(1536) == "1.5 KB" and v.size_text(12 * 1024 ** 3) == "12.0 GB" and v.size_text(None) == "-"
    assert v.duration_text(30) == "under 2 min" and v.duration_text(4000) == "1 h 07 min"


def test_cpu_cores_have_one_decimal_and_memory_is_in_gb_as_cards_are_sold():
    o = mk.Offer(id=1, gpu_total_ram_mb=32607, cpu_cores=14.2222, cpu_ram_mb=131072, cpu_name="")
    row = v.offer_row(o)
    assert row["vram"] == "32 GB" and row["cpu"] == "14.2 cores" and row["ram"] == "128 GB"
    assert v.offer_row(mk.Offer(id=1, gpu_total_ram_mb=24564, cpu_cores=64.0))["cpu"] == "64 cores"
    assert v.offer_row(mk.Offer(id=1, gpu_total_ram_mb=24564))["vram"] == "24 GB"


def test_the_filter_groups_have_their_new_names():
    assert "Rental, GPU and region" in mk.GROUPS and "More" in mk.GROUPS
    assert "Top row" not in mk.GROUPS and "Also" not in mk.GROUPS
    assert {f.group for f in mk.FILTERS} <= set(mk.GROUPS)


def test_billing_is_only_from_a_page_opened_on_this_computer(monkeypatch):
    notes = []
    monkeypatch.setattr(v.ui, "notify", lambda text, **kw: notes.append(text))
    away = v.VastUI(SimpleNamespace(local=False), services=FakeServices(), watch=None)
    here = v.VastUI(SimpleNamespace(local=True), services=FakeServices(), watch=None)
    assert here.guard_local() is True and notes == []
    assert away.guard_local() is False and notes == ["Only from a browser on this computer."]
    assert v.VastUI(SimpleNamespace(), services=FakeServices(), watch=None).local is True  # no word: this computer


def test_user_text_says_rulz_ai_trainer():
    assert "RulZ Trainer" not in SRC


def test_unconfirmed_stops_show_as_alerts_until_dismissed():
    now = time.time()

    class Manager:
        def sends(self):
            return [{"warnings": ["Stopping instance 7 failed: x It still bills: stop it on vast.ai now.", "Other."]}]

    class Fleet:
        def events(self, kinds=None):
            return [{"kind": "send_alert", "time": now - 60, "error": "Instance 8 still billing."},
                    {"kind": "send_alert", "time": now - 7200, "error": "old"}]

    services = FakeServices(manager=Manager())
    services.manager, services.fleet_getter = Manager(), lambda: Fleet()
    page = v.VastUI(SimpleNamespace(), services=services, watch=None)
    page.notices = page.collect_notices()
    assert page.notices == ["Stopping instance 7 failed: x It still bills: stop it on vast.ai now.",
                            "Instance 8 still billing."]
    alerts = page.merged({"alerts": []})["alerts"]
    assert [a["text"] for a in alerts] == page.notices
    assert v.chip_view({**snap(), "alerts": alerts})["level"] == "critical"
    page.dismissed.add(alerts[0]["id"])
    assert len(page.merged({"alerts": []})["alerts"]) == 1
    assert len(page.merged(page.merged({"alerts": []}))["alerts"]) == 1  # merging twice adds nothing


# ---------------------------------------------------------------- review findings: one dialog, fresh start, stale filters


def test_a_second_click_while_a_dialog_is_open_does_nothing():
    import asyncio

    page = v.VastUI(SimpleNamespace(root=REPO), services=FakeServices(), watch=None)
    page.sending_busy = True
    assert asyncio.run(page.open_send({}, "job")) is None  # nothing is built, nothing raises
    page.rent_busy = True
    assert asyncio.run(page.open_rent(object(), {})) is None
    assert page.sending_busy is True and page.rent_busy is True  # the open ones keep their place


def test_the_page_takes_its_place_before_the_first_await():
    page = Path(webui.__file__).read_text(encoding="utf-8")
    body = page.split("async def send_job(")[1].split("async def start_or_send")[0]
    assert body.index("vast_ui.sending_busy = True") < body.index("await run.io_bound(start_warnings")
    assert "finally:\n            vast_ui.sending_busy = False" in body and "held=True" in body


def test_a_start_is_asked_again_when_the_status_or_the_price_moved():
    shown = {"id": 5, "status": "exited", "dph_total": 0.61}
    assert v.instance_changed(shown, {"id": 5, "status": "exited", "dph_total": 0.61}) == ""
    assert "now running" in v.instance_changed(shown, {"id": 5, "status": "running", "dph_total": 0.61})
    assert v.instance_changed(shown, {"id": 5, "status": "exited", "dph_total": 0.64}) == \
        "The price changed from $0.61/hour to $0.64/hour."
    assert "not on your account" in v.instance_changed(shown, None)


def test_create_gets_the_shown_price_only_when_the_manager_takes_it():
    def old(job, config, root, *, instance=None, start=False): ...
    def new(job, config, root, *, instance=None, start=False, shown=None): ...
    def kw(job, config, root, **kw): ...
    assert not v.accepts(old, "shown") and v.accepts(new, "shown") and v.accepts(kw, "shown")


def test_elements_that_are_gone_are_not_touched():
    assert v.alive(None, SimpleNamespace(is_deleted=False)) is True
    assert v.alive(SimpleNamespace(is_deleted=True)) is False
    assert v.alive(SimpleNamespace()) is True  # an element without the flag is taken as there
    dlg = v.SendDialog.__new__(v.SendDialog)
    dlg.closed, dlg.dlg = False, SimpleNamespace(is_deleted=False)
    assert dlg.live() is True
    dlg.closed = True
    assert dlg.live() is False
    dlg.closed, dlg.dlg = False, SimpleNamespace(is_deleted=True)
    assert dlg.live() is False


def test_a_notification_from_a_gone_page_does_not_raise(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("the client was deleted")

    monkeypatch.setattr(v.ui, "notify", boom)
    v.notify_page("sent", type="positive")  # no exception
    assert "ui.notify(" not in SRC.replace("        ui.notify(text, **kw)", "")


def test_the_rent_uses_the_filters_the_offers_were_searched_with():
    class Fake:
        def __init__(self, searched, now):
            self.searched_with, self.now = searched, now
            self.stale = SimpleNamespace(visible=False)

        def cleaned(self):
            return self.now

    searched = {"filters": {"disk": 200}, "template": {"hash_id": "t1"}}
    same, moved, never = Fake(searched, {"disk": 200}), Fake(searched, {"disk": 400}), Fake(None, {"disk": 400})
    for fake in (same, moved, never):
        v.MarketView.paint_stale(fake)
    assert same.stale.visible is False and moved.stale.visible is True and never.stale.visible is False
    src = SRC.split("async def on_rent")[1].split("# ----")[0]
    assert 'self.searched_with["filters"]' in src and "self.read()" not in src
    assert "Filters changed: search again to apply them." in SRC


def test_cancel_turns_into_close_once_the_send_is_on_its_way():
    go = SRC.split("    async def go(self)")[1].split("    def show_progress")[0]
    assert go.index('self.cancel_btn.text = "Close"') < go.index("manager.create")
    assert "The send goes on in the background" in go and "notify_page(" in go and "if not self.live()" in go
    run_ = SRC.split("    async def run(self)")[1].split("    def live")[0]
    assert "self.timer.cancel()" in run_ and run_.index("self.closed = True") < run_.index("self.dlg.delete()")


# ---------------------------------------------------------------- who may bill


@pytest.mark.parametrize("header", ["tailscale-user-login", "x-forwarded-for", "x-real-ip", "forwarded",
                                    "x-forwarded-host", "x-forwarded-proto", "cf-connecting-ip", "via"])
def test_a_request_that_came_through_a_proxy_is_not_a_local_viewer(header):
    import webui
    here = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"), headers={"host": "localhost:7860",
                                                                              "user-agent": "Mozilla/5.0"})
    assert webui.is_local_viewer(here) is True  # a browser on this PC sends none of the proxy headers
    proxied = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"), headers={header: "x"})
    assert webui.is_local_viewer(proxied) is False
    assert webui.is_local_viewer(SimpleNamespace(client=SimpleNamespace(host="192.168.1.5"), headers={})) is False


def test_a_browser_that_is_not_local_cannot_sign_in_create_a_bucket_or_pick_destroy(monkeypatch):
    import asyncio
    warned = []
    monkeypatch.setattr(v, "notify_page", lambda text, **kw: warned.append(text))
    touched = []
    vast = SimpleNamespace(local=False, services=SimpleNamespace(module=lambda name: touched.append(name)),
                           guard_local=lambda: (warned.append(v.LOCAL_ONLY), False)[1])
    dialog = SimpleNamespace(vast=vast, token_input=SimpleNamespace(value="hf_secret"), bucket="me/mine", after="stop",
                             after_touched=False)
    asyncio.run(v.SendDialog.sign_in(dialog))
    asyncio.run(v.SendDialog.create_bucket(dialog))
    assert touched == [] and dialog.token_input.value == "hf_secret" and len(warned) == 2
    sender = SimpleNamespace(value="destroy")
    v.SendDialog.set_after(dialog, SimpleNamespace(value="destroy", sender=sender))
    assert dialog.after == "stop" and sender.value == "stop" and v.LOCAL_ONLY in warned
