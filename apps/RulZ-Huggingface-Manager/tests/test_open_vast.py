import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rulz_hf_manager import vast
from rulz_hf_manager.ui import vast_page


def test_removing_stale_basket_item_keeps_other_items(monkeypatch):
    first = vast.Source('model', 'owner/first')
    second = vast.Source('model', 'owner/second')
    third = vast.Source('model', 'owner/third')
    monkeypatch.setattr(vast_page, 'BASKET', [first, second, third])
    vast_page.remove_from_basket(first)
    vast_page.remove_from_basket(second)
    vast_page.remove_from_basket(second)
    assert vast_page.BASKET == [third]


def test_stale_basket_item_does_not_remove_readded_equal_item(monkeypatch):
    stale = vast.Source('model', 'owner/model')
    replacement = vast.Source('model', 'owner/model')
    monkeypatch.setattr(vast_page, 'BASKET', [replacement])
    vast_page.remove_from_basket(stale)
    assert vast_page.BASKET == [replacement]


# ---- per-tick redraws: update in place, rebuild only when the shape changes -----------------------------------------
import asyncio  # noqa: E402
import time  # noqa: E402

from nicegui import Client, ui  # noqa: E402
from nicegui.element import Element  # noqa: E402

from rulz_hf_manager.jobs import Job, manager  # noqa: E402
from rulz_hf_manager.ui import common  # noqa: E402


def _count_elements(monkeypatch) -> list:
    created = [0]
    orig = Element.__init__

    def counting(self, *a, **k):
        created[0] += 1
        orig(self, *a, **k)

    monkeypatch.setattr(Element, '__init__', counting)
    return created


def _transfers(step: int, state: str = 'running') -> list[dict]:
    return [{'id': f't{i}', 'title': f'model {i}', 'state': state, 'bytes_total': 10**10,
             'bytes_done': step * 10**8, 'files_done': step, 'files_total': 20, 'speed': 5e7,
             'dest': '/workspace/x'} for i in range(5)]


def test_transfer_cards_update_in_place_when_only_progress_changes(monkeypatch):
    created = _count_elements(monkeypatch)
    with Client(ui.page('/t-cards')):
        cards = vast_page.TransferCards(ui.column(), lambda *a: None, lambda s: None, lambda s: None, lambda s: None)
        assert cards.show(_transfers(1)) is True
        before = created[0]
        assert before > 5
        bar = cards.refs['t0']['bar']
        assert cards.show(_transfers(7)) is False
        assert created[0] == before
        assert cards.refs['t0']['bar'] is bar
        assert bar.value == 0.07  # 7e8 of 1e10 bytes
        assert cards.refs['t0']['line'].text.startswith('7/20 files')
        assert cards.show(_transfers(8, 'done')) is True  # a state change rebuilds
        assert created[0] > before


def test_activity_drawer_updates_running_job_in_place(monkeypatch):
    monkeypatch.setattr(manager, 'jobs', [])
    for i in range(1, 11):
        job = Job(id=i, title=f'job {i}', status='done')
        job.started, job.finished = time.time() - 5, time.time()
        manager.jobs.append(job)
    running = Job(id=50, title='running', status='running')
    running.started, running.progress = time.time() - 3, 0.1
    manager.jobs.append(running)
    created = _count_elements(monkeypatch)

    async def go() -> None:
        with Client(ui.page('/t-drawer')):
            refs: dict = {}
            view = ui.refreshable(common._job_list)
            view(refs)
            shape, before, bar = common._jobs_shape(), created[0], refs[50]['bar']
            assert before > 10
            running.progress = 0.6
            assert common._jobs_shape() == shape  # progress alone does not change the shape
            common._update_job_cards(refs)
            assert created[0] == before
            assert refs[50]['bar'] is bar and bar.value == 0.6
            running.status = 'done'
            assert common._jobs_shape() != shape  # a status change does

    asyncio.run(go())
