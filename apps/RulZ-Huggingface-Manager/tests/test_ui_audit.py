from pathlib import Path

from rulz_hf_manager.ui import browser, vast_page


def test_shell_batch_keeps_only_latest_lines():
    assert vast_page._tail_shell_lines(['a', 'b', 'c'], 2) == ['b', 'c']
    assert vast_page._tail_shell_lines(['a'], 2) == ['a']


def test_download_sweep_respects_six_hour_age(monkeypatch):
    old, recent = Path('old'), Path('recent')
    entries = [(old, 0), (recent, 1)]
    removed = []
    monkeypatch.setattr(browser, '_temp_downloads', entries)
    monkeypatch.setattr(browser.hub, 'cleanup_temp', removed.append)
    browser._sweep_downloads(6 * 60 * 60)
    assert removed == [old]
    assert entries == [(recent, 1)]


def test_markdown_limit_counts_utf8_bytes():
    assert browser._render_markdown('a' * (200 * 1024))
    assert not browser._render_markdown('a' * (200 * 1024 + 1))
    assert not browser._render_markdown('é' * (100 * 1024 + 1))


def test_markdown_warm_up_fills_the_cache_ui_markdown_reads():
    # ui.markdown converts through prepare_content(content, extras=' '.join(extras)); the warm-up in a worker thread
    # only helps if the event loop's later call is a cache hit.
    from nicegui.elements.markdown import prepare_content
    text = '# audit warm-up check\n\n```py\nx = 1\n```\n'
    browser._warm_markdown(text)
    hits = prepare_content.cache_info().hits
    prepare_content(text, extras=' '.join(browser.MARKDOWN_EXTRAS))
    assert prepare_content.cache_info().hits == hits + 1
