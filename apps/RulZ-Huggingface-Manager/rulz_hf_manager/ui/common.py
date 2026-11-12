"""Shared UI building blocks: page frame, theme, dialogs, formatting helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

from nicegui import Client, run, ui

from .. import APP_NAME, config, hub
from ..jobs import Job, manager
from ..platform_utils import open_folder
from .theme import LOGO_DARK, LOGO_LIGHT, PageLook, choose_look

log = logging.getLogger('rulz_hf_manager')

# --------------------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------------------

def fmt_size(n: int | float | None) -> str:
    return hub.human_size(n) if n is not None else ''


def fmt_date(dt: datetime | None) -> str:
    if not dt:
        return ''
    if dt.tzinfo is not None:
        dt = dt.astimezone()
    return dt.strftime('%Y-%m-%d %H:%M')


def fmt_ago(dt: datetime | None) -> str:
    if not dt:
        return ''
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    s = (now - dt).total_seconds()
    for limit, div, unit in ((60, 1, 's'), (3600, 60, 'min'), (86400, 3600, 'h'), (86400 * 30, 86400, 'd'),
                             (86400 * 365, 86400 * 30, 'mo')):
        if s < limit:
            return f'{int(s // div)} {unit} ago'
    return f'{int(s // (86400 * 365))} y ago'


def fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f'{seconds}s'
    if seconds < 3600:
        return f'{seconds // 60}m {seconds % 60:02d}s'
    return f'{seconds // 3600}h {(seconds % 3600) // 60:02d}m'


SIZE_FORMAT_JS = ('(v, row) => (row && row.is_dir) || v === null || v === undefined ? "" : '
                  '(v < 1024 ? v + " B" : v < 1048576 ? (v/1024).toFixed(1) + " KB" : '
                  'v < 1073741824 ? (v/1048576).toFixed(1) + " MB" : v < 1099511627776 ? '
                  '(v/1073741824).toFixed(2) + " GB" : (v/1099511627776).toFixed(2) + " TB")')


def browse_url(kind: str, repo_id: str, path: str = '', rev: str | None = None) -> str:
    url = f'/browse/{kind}/{quote(repo_id)}'
    params = {k: v for k, v in (('path', path.strip('/')), ('rev', rev or '')) if v}
    return f'{url}?{urlencode(params)}' if params else url


# --------------------------------------------------------------------------------------------------
# Notifications & client-safe callbacks
# --------------------------------------------------------------------------------------------------

def notify_error(e: BaseException | str, prefix: str = '') -> None:
    msg = e if isinstance(e, str) else hub.friendly_error(e)
    ui.notify(f'{prefix}{msg}', type='negative', multi_line=True, close_button=True, timeout=10000)


def notify_ok(msg: str) -> None:
    ui.notify(msg, type='positive')


def in_client(client: Client, fn: Callable[[], Any]) -> None:
    """Run UI code for a specific browser tab from a background callback (skips closed tabs)."""
    try:
        if client.is_deleted or not client.has_socket_connection:
            return
        with client:
            fn()
    except Exception:
        log.debug('UI callback skipped', exc_info=True)


def run_job(title: str, fn: Callable[[Job], Any], *, icon: str = 'sync',
            success: str | Callable[[Job], str] | None = None,
            after: Callable[[Job], Any] | None = None) -> Job:
    """Submit a background job and notify the current tab when it finishes."""
    client = ui.context.client
    ui.notify(f'Started: {title}', type='ongoing', timeout=2500)

    def ok(job: Job) -> None:
        def show() -> None:
            text = success(job) if callable(success) else (success or f'Done: {title}')
            ui.notify(text, type='positive', multi_line=True)
            if after:
                after(job)
        in_client(client, show)

    def failed(job: Job) -> None:
        in_client(client, lambda: notify_error(job.error, prefix=f'{title} failed: '))

    return manager.submit(title, fn, icon=icon, on_success=ok, on_error=failed)


# --------------------------------------------------------------------------------------------------
# Dialogs
# --------------------------------------------------------------------------------------------------

async def confirm(title: str, message: str = '', *, ok_label: str = 'OK', ok_color: str = 'primary',
                  items: list[str] | None = None, require_text: str | None = None, warning: str = '') -> bool:
    with ui.dialog() as dialog, ui.card().classes('rulz-card w-[560px] max-w-[95vw]'):
        ui.label(title).classes('text-lg font-semibold')
        if message:
            ui.label(message).classes('whitespace-pre-line')
        if items:
            with ui.scroll_area().classes('max-h-48 w-full rz-box'):
                for it in items[:200]:
                    ui.label(it).classes('rulz-mono text-sm')
                if len(items) > 200:
                    ui.label(f'... and {len(items) - 200} more').classes('text-sm rulz-muted')
        if warning:
            with ui.row().classes('items-center gap-2 rz-warn no-wrap'):
                ui.icon('warning')
                ui.label(warning).classes('text-sm')
        typed = None
        if require_text:
            ui.label(f'Type "{require_text}" to confirm:').classes('text-sm')
            typed = ui.input().props('dense outlined autofocus').classes('w-full rulz-mono')
        with ui.row().classes('w-full justify-end gap-2'):
            ui.button('Cancel', on_click=lambda: dialog.submit(False)).props('flat')
            ok = ui.button(ok_label, color=ok_color, on_click=lambda: dialog.submit(True))
            if typed is not None:
                ok.bind_enabled_from(typed, 'value', backward=lambda v: v.strip() == require_text)
    return bool(await dialog)


async def prompt(title: str, label: str = '', value: str = '', *, ok_label: str = 'OK', hint: str = '',
                 validation: Callable[[str], str | None] | None = None, select_stem: bool = False) -> str | None:
    with ui.dialog() as dialog, ui.card().classes('rulz-card w-[520px] max-w-[95vw]'):
        ui.label(title).classes('text-lg font-semibold')
        inp = ui.input(label, value=value).props('outlined autofocus').classes('w-full')
        if hint:
            ui.label(hint).classes('text-xs rulz-muted')
        error = ui.label().classes('rz-err text-sm')

        def submit() -> None:
            v = (inp.value or '').strip()
            msg = validation(v) if validation else None
            if msg:
                error.text = msg
                return
            dialog.submit(v)
        inp.on('keydown.enter', submit)
        with ui.row().classes('w-full justify-end gap-2'):
            ui.button('Cancel', on_click=lambda: dialog.submit(None)).props('flat')
            ui.button(ok_label, on_click=submit)
    if select_stem and '.' in value.lstrip('.'):
        stem_len = value.rfind('.')
        ui.timer(0.25, lambda: ui.run_javascript(
            f'const el = getHtmlElement({inp.id})?.querySelector("input"); '
            f'if (el) {{ el.focus(); el.setSelectionRange(0, {stem_len}); }}'), once=True)
    return await dialog


def commit_fields(fs: hub.RemoteFS, default_message: str) -> tuple[ui.input | None, ui.checkbox | None]:
    """Commit message + 'open as pull request' inputs (only meaningful for git repos)."""
    if fs.is_bucket:
        ui.label('Buckets have no history: changes apply immediately.').classes('text-xs rulz-muted')
        return None, None
    msg = ui.input('Commit message', value=default_message).props('outlined dense').classes('w-full')
    pr = ui.checkbox('Open as a pull request instead of committing directly')
    return msg, pr


# --------------------------------------------------------------------------------------------------
# Page frame
# --------------------------------------------------------------------------------------------------

async def open_location(text: str) -> None:
    try:
        kind, repo_id, path = hub.parse_location(text)
        if kind is None:
            kind = await run.io_bound(hub.detect_kind, repo_id)
        ui.navigate.to(browse_url(kind, repo_id, path))
    except Exception as e:  # noqa: BLE001
        notify_error(e)


def _job_list() -> None:
    if not manager.jobs:
        with ui.column().classes('w-full items-center py-10 rulz-muted'):
            ui.icon('inbox', size='lg')
            ui.label('No activity yet')
        return
    for job in manager.jobs:
        with ui.card().classes('w-full rulz-card rz-job p-3 gap-1').props('flat'):
            with ui.row().classes('w-full items-center no-wrap gap-2'):
                if job.status == 'running':
                    ui.spinner(size='sm', color='primary')
                elif job.status == 'queued':
                    ui.icon('schedule').classes('rz-ic-muted')
                else:
                    ui.icon('check_circle' if job.status == 'done' else 'error',
                            color='positive' if job.status == 'done' else 'negative')
                ui.label(job.title).classes('text-sm font-medium flex-1 ellipsis')
                ui.label(fmt_duration(job.elapsed) if job.started else 'queued').classes('text-xs rulz-muted')
            if job.status == 'running':
                if job.progress is not None:
                    ui.linear_progress(value=job.progress, show_value=False).props('rounded size=6px')
                else:
                    ui.linear_progress(show_value=False).props('indeterminate rounded size=6px')
            if job.message or job.detail:
                ui.label(' / '.join(x for x in (job.message, job.detail) if x)).classes('text-xs rulz-muted ellipsis')
            if job.error:
                ui.label(job.error).classes('text-xs rz-err whitespace-pre-line')
            if job.status == 'done' and isinstance(job.result, Path) and job.result.is_dir():
                result = job.result
                with ui.row().classes('items-center gap-1'):
                    ui.label(str(result)).classes('text-xs rulz-mono rulz-muted ellipsis max-w-[250px]')
                    ui.button('Open folder', icon='folder_open',
                              on_click=lambda r=result: open_folder(r) or ui.notify('Could not open the folder')
                              ).props('flat dense size=sm no-caps')


@contextmanager
def frame(active: str = '') -> Iterator[PageLook]:
    """Header + activity drawer + centered content column. Yields the tab's look (dark or light)."""
    look = PageLook()
    ui.page_title(APP_NAME)

    with ui.right_drawer(value=False, fixed=True).props('width=400 bordered').classes('p-3 gap-3') as drawer:
        with ui.row().classes('w-full items-center no-wrap gap-2'):
            ui.icon('pending_actions').classes('rz-sec-icon')
            ui.label('Activity').classes('rz-title')
            ui.space()
            ui.button('Clear finished', on_click=manager.clear_finished, color=None) \
                .props('flat dense no-caps size=sm').classes('rz-icon-btn')
            ui.button(icon='close', on_click=drawer.hide, color=None).props('flat round dense').classes('rz-icon-btn')
        jobs_view = ui.refreshable(_job_list)
        jobs_view()

    with ui.header().classes('rz-header'):
        with ui.row().classes('rz-header-row items-center gap-2 no-wrap'):
            with ui.link(target='/').classes('rz-brand row items-center gap-3 no-wrap'):
                ui.element('img').props(f'src="{LOGO_DARK}" alt="RulZ"').classes('rz-logo rz-dark-only')
                ui.element('img').props(f'src="{LOGO_LIGHT}" alt="RulZ"').classes('rz-logo rz-light-only')
                ui.label('HUGGINGFACE MANAGER').classes('rz-wordmark gt-xs')
            ui.element('div').classes('w-3')
            for label, icon, target, key in (('My Hub', 'space_dashboard', '/', 'hub'),
                                             ('vast.ai', 'dns', '/vast', 'vast'),
                                             ('Settings', 'settings', '/settings', 'settings')):
                btn = ui.button(label, icon=icon, on_click=lambda t=target: ui.navigate.to(t), color=None) \
                    .props('flat no-caps dense').classes('rz-nav-btn gt-sm')
                if key == active:
                    btn.classes('rz-on')
            ui.space()
            open_input = ui.input(placeholder='Open repo / bucket / URL ...') \
                .props('dense outlined clearable').classes('rz-open-input w-[340px] max-w-[40vw]')
            with open_input.add_slot('prepend'):
                ui.icon('search')
            open_input.on('keydown.enter', lambda: open_location(open_input.value or ''))
            with ui.button(icon='pending_actions', on_click=drawer.toggle, color=None) \
                    .props('flat round').classes('rz-icon-btn'):
                badge = ui.badge('', color='primary').props('floating rounded')
                ui.tooltip('Activity (uploads, downloads, moves...)')
            # The sun shows in the dark look (switch to light), the moon in the light look. CSS picks the icon,
            # so it is right from the first paint.
            with ui.button(on_click=look.toggle, color=None).props('flat round').classes('rz-icon-btn'):
                ui.icon('light_mode').classes('rz-dark-only')
                ui.icon('dark_mode').classes('rz-light-only')
                ui.tooltip('Switch between the dark and the light look')
            avatar_slot = ui.row().classes('items-center')

    state = {'version': -1, 'active': 0}

    def tick() -> None:
        if manager.version != state['version']:
            state['version'] = manager.version
            jobs_view.refresh()
            n = manager.active_count
            badge.set_text(str(n) if n else '')
            badge.set_visibility(bool(n))
            if n > state['active'] and not drawer.value:
                drawer.show()
            state['active'] = n
        elif manager.active_count:
            jobs_view.refresh()  # keep elapsed time ticking
    ui.timer(0.7, tick)

    async def load_identity() -> None:
        try:
            ident = await run.io_bound(hub.whoami)
        except Exception:  # noqa: BLE001
            ident = None
        avatar_slot.clear()
        with avatar_slot:
            if ident:
                with ui.button(color=None).props('flat round dense').classes('rz-icon-btn'):
                    if ident.avatar:
                        ui.image(ident.avatar).classes('w-8 h-8 rounded-full')
                    else:
                        ui.icon('account_circle', size='md')
                    with ui.menu():
                        with ui.column().classes('p-3 gap-0'):
                            ui.label(ident.fullname or ident.name).classes('font-semibold')
                            ui.label(f'@{ident.name}').classes('text-xs rulz-muted')
                            if ident.token_role:
                                ui.label(f'Token: {ident.token_name or "unnamed"} ({ident.token_role})') \
                                    .classes('text-xs rulz-muted')
                            elif ident.is_oauth:
                                until = f' until {ident.expires:%d %b %Y}' if ident.expires else ''
                                ui.label(f'Login token (OAuth){until}').classes('text-xs rulz-muted')
                        ui.separator()
                        ui.menu_item('Settings', on_click=lambda: ui.navigate.to('/settings'))
                        ui.menu_item('Open my profile on huggingface.co',
                                     on_click=lambda: ui.navigate.to(f'{hub.endpoint()}/{ident.name}', new_tab=True))
            else:
                ui.button('Add token', icon='key', on_click=lambda: ui.navigate.to('/settings')) \
                    .props('unelevated color=primary no-caps dense').classes('px-3')
    ui.timer(0.01, load_identity, once=True)

    if not config.get().look_chosen:
        ui.timer(0.2, lambda: choose_look(look), once=True)

    with ui.column().classes('w-full max-w-[1400px] mx-auto p-4 gap-4'):
        yield look
