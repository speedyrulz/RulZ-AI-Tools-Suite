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

log = logging.getLogger('rulz_hf_manager')

LOGO_SVG = '''
<svg width="34" height="34" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <defs><linearGradient id="rg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#FBBF24"/><stop offset="1" stop-color="#F97316"/></linearGradient></defs>
  <rect x="2" y="2" width="60" height="60" rx="16" fill="url(#rg)"/>
  <path d="M20 46V18h13.5c6.4 0 10.5 3.6 10.5 9.1 0 4.1-2.3 7.1-6 8.3L45 46h-7.6l-6.3-10H27v10h-7zm7-16h6.2c2.6 0 4.1-1.3 4.1-3.4s-1.5-3.4-4.1-3.4H27V30z" fill="#1E1B4B"/>
</svg>
'''

CSS = '''
:root { --rulz-radius: 14px; }
.body--light { background: #f4f5fb; }
.body--dark { background: #0c0f16; }
.rulz-header { background: linear-gradient(90deg, #1e1b4b 0%, #3730a3 50%, #6d28d9 100%) !important; }
.rulz-card { border-radius: var(--rulz-radius) !important; }
.body--dark .rulz-card, .body--dark .q-table__card { background: #151a24 !important; }
.q-table__card { border-radius: var(--rulz-radius) !important; }
.rulz-table thead tr th { font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .03em; opacity: .75; }
.rulz-table tbody tr { cursor: default; }
.rulz-table .rulz-name:hover span { text-decoration: underline; }
.rulz-muted { opacity: .65; }
.rulz-crumb .q-btn { text-transform: none; }
.rulz-open-input .q-field__control { background: rgba(255,255,255,.12); border-radius: 10px; }
.rulz-open-input input { color: white !important; }
.rulz-open-input input::placeholder { color: rgba(255,255,255,.7) !important; }
.rulz-drop .q-uploader { width: 100%; }
.rulz-mono { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
.rulz-stat { min-width: 150px; }
'''

DARK_VALUE = {'auto': None, 'light': False, 'dark': True}


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
            with ui.scroll_area().classes('max-h-48 w-full border rounded'):
                for it in items[:200]:
                    ui.label(it).classes('rulz-mono text-sm')
                if len(items) > 200:
                    ui.label(f'... and {len(items) - 200} more').classes('text-sm rulz-muted')
        if warning:
            with ui.row().classes('items-center gap-2 text-orange-8 no-wrap'):
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
        error = ui.label().classes('text-negative text-sm')

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

def apply_theme() -> ui.dark_mode:
    ui.colors(primary='#4F46E5', secondary='#F59E0B', accent='#DB2777', positive='#16A34A',
              negative='#DC2626', warning='#F59E0B', info='#0EA5E9')
    ui.add_css(CSS)
    return ui.dark_mode(DARK_VALUE.get(config.get().dark_mode))


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
        color = {'done': 'positive', 'error': 'negative', 'running': 'primary', 'queued': 'grey'}[job.status]
        with ui.card().classes('w-full rulz-card p-3 gap-1').props('flat bordered'):
            with ui.row().classes('w-full items-center no-wrap gap-2'):
                if job.status == 'running':
                    ui.spinner(size='sm', color='primary')
                else:
                    ui.icon({'done': 'check_circle', 'error': 'error', 'queued': 'schedule'}[job.status], color=color)
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
                ui.label(job.error).classes('text-xs text-negative whitespace-pre-line')
            if job.status == 'done' and isinstance(job.result, Path) and job.result.is_dir():
                result = job.result
                with ui.row().classes('items-center gap-1'):
                    ui.label(str(result)).classes('text-xs rulz-mono rulz-muted ellipsis max-w-[250px]')
                    ui.button('Open folder', icon='folder_open',
                              on_click=lambda r=result: open_folder(r) or ui.notify('Could not open the folder')
                              ).props('flat dense size=sm no-caps')


@contextmanager
def frame(active: str = '') -> Iterator[None]:
    """Header + activity drawer + centered content column."""
    apply_theme()
    ui.page_title(APP_NAME)

    with ui.right_drawer(value=False, fixed=True).props('width=400 bordered').classes('p-3 gap-3') as drawer:
        with ui.row().classes('w-full items-center'):
            ui.icon('pending_actions', size='sm')
            ui.label('Activity').classes('text-lg font-semibold')
            ui.space()
            ui.button('Clear finished', on_click=manager.clear_finished).props('flat dense no-caps size=sm')
            ui.button(icon='close', on_click=drawer.hide).props('flat round dense')
        jobs_view = ui.refreshable(_job_list)
        jobs_view()

    with ui.header(elevated=True).classes('rulz-header items-center px-4 py-2 gap-2 no-wrap'):
        with ui.link(target='/').classes('row items-center gap-2 no-underline text-white no-wrap'):
            ui.html(LOGO_SVG, sanitize=False)
            with ui.column().classes('gap-0 leading-none gt-xs'):
                ui.label('RulZ').classes('text-lg font-black tracking-wide leading-none')
                ui.label('Huggingface Manager').classes('text-xs font-light opacity-90 leading-none')
        ui.element('div').classes('w-4')
        for label, icon, target, key in (('My Hub', 'space_dashboard', '/', 'hub'),
                                         ('Settings', 'settings', '/settings', 'settings')):
            btn = ui.button(label, icon=icon, on_click=lambda t=target: ui.navigate.to(t)) \
                .props('flat no-caps color=white').classes('gt-sm')
            if key == active:
                btn.classes('bg-white/20')
        ui.space()
        open_input = ui.input(placeholder='Open repo / bucket / URL ...') \
            .props('dense borderless clearable dark').classes('rulz-open-input w-[340px] max-w-[45vw] px-2')
        with open_input.add_slot('prepend'):
            ui.icon('search', color='white')
        open_input.on('keydown.enter', lambda: open_location(open_input.value or ''))
        with ui.button(icon='pending_actions', on_click=drawer.toggle).props('flat round color=white'):
            badge = ui.badge('', color='orange').props('floating rounded')
            ui.tooltip('Activity (uploads, downloads, moves...)')
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
                with ui.button().props('flat round dense'):
                    if ident.avatar:
                        ui.image(ident.avatar).classes('w-8 h-8 rounded-full')
                    else:
                        ui.icon('account_circle', color='white', size='md')
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
                    .props('unelevated color=orange no-caps dense')
    ui.timer(0.01, load_identity, once=True)

    with ui.column().classes('w-full max-w-[1400px] mx-auto p-4 gap-4'):
        yield
