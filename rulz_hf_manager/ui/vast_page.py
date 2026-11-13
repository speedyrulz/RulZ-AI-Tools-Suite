"""vast.ai page: attach an instance, collect models, send them straight from Hugging Face to the instance.

Also holds the "Send to vast.ai" dialog that My Hub and the file browser open.
"""

from __future__ import annotations

import asyncio
import posixpath
import queue
import threading
import time

from nicegui import run, ui

from .. import config, hub, vast
from ..jobs import Job
from ..remote_agent import COMFY_DIRS
from .common import browse_url, confirm, fmt_duration, frame, notify_error, notify_ok, prompt, run_job

# What the user collected to send together (lives as long as the app runs).
BASKET: list[vast.Source] = []
_info: dict = {'conn': None, 'info': None, 'at': 0.0}
# Command prompt history and each instance's current folder (as long as the app runs).
SHELL: dict = {'history': [], 'cwd': {}}

STATE_LOOK = {
    'queued': ('schedule', 'grey'), 'starting': ('hourglass_top', 'grey'), 'installing': ('build', 'warning'),
    'running': ('downloading', 'primary'), 'done': ('check_circle', 'positive'), 'error': ('error', 'negative'),
    'cancelled': ('block', 'grey'), 'stopped': ('pause_circle', 'warning'),
}


async def gather_in(element: ui.element, *coros) -> list:
    """asyncio.gather for UI code: each coroutine runs in a task of its own, which starts without the page's
    context, so enter ``element`` in each (otherwise ui.notify and friends fail with "slot stack is empty")."""
    async def inside(coro):
        with element:
            return await coro
    return await asyncio.gather(*(inside(c) for c in coros))


def add_to_basket(sources: list[vast.Source]) -> None:
    added = 0
    for s in sources:
        if not any(b.to_dict() == s.to_dict() for b in BASKET):
            BASKET.append(s)
            added += 1
    ui.notify(f'Added {added} to the vast.ai transfer list ({len(BASKET)} in it)' if added
              else 'Already in the vast.ai transfer list', type='positive' if added else 'info')


async def instance_info(force: bool = False) -> dict | None:
    """The attached instance's info (folders worth a click, disk...), cached for a minute."""
    conn = vast.attached()
    if not conn:
        return None
    key = (conn.host, conn.port)
    if not force and _info['conn'] == key and _info['info'] and time.time() - _info['at'] < 60:
        return _info['info']
    info = await run.io_bound(lambda: vast.Remote(conn).call('info', timeout=60))
    _info.update(conn=key, info=info, at=time.time())
    return info


# --------------------------------------------------------------------------------------------------
# Remote folder picker
# --------------------------------------------------------------------------------------------------

async def pick_vast_folder(start: str = '/workspace', title: str = 'Choose a folder on the instance') -> str | None:
    state = {'path': start or '/workspace', 'hidden': False}

    with ui.dialog() as dialog, ui.card().classes('rulz-card w-[680px] max-w-[95vw] h-[75vh] p-0 gap-0 column no-wrap'):
        with ui.row().classes('w-full items-center px-4 py-3 gap-2 no-wrap'):
            ui.icon('dns', size='sm', color='primary')
            ui.label(title).classes('text-lg font-semibold')
            ui.space()
            ui.button(icon='close', on_click=lambda: dialog.submit(None)).props('flat round dense')
        ui.separator()
        with ui.row().classes('w-full items-center px-3 py-2 gap-1 no-wrap'):
            up_btn = ui.button(icon='arrow_upward', on_click=lambda: go(posixpath.dirname(state['path']) or '/')) \
                .props('flat round dense').tooltip('Up one folder')
            path_input = ui.input(value=state['path']).props('dense outlined').classes('flex-1 rulz-mono')
            path_input.on('keydown.enter', lambda: go(path_input.value or '/'))
            ui.checkbox('Hidden', on_change=lambda e: (state.update(hidden=e.value), go(state['path']))) \
                .props('dense')
        listing = ui.column().classes('w-full flex-1 overflow-auto px-2 gap-0')
        ui.separator()
        with ui.row().classes('w-full items-center px-4 py-3 gap-2'):
            ui.button('New folder', icon='create_new_folder', on_click=lambda: new_folder()).props('flat no-caps')
            free = ui.label().classes('text-xs rulz-muted')
            ui.space()
            ui.button('Cancel', on_click=lambda: dialog.submit(None)).props('flat')
            choose = ui.button('Choose', icon='check', on_click=lambda: dialog.submit(state['path']))

    async def new_folder() -> None:
        name = await prompt('New folder', 'Folder name',
                            hint='Created when the first file arrives.',
                            validation=lambda v: None if v and '/' not in v and v not in {'.', '..'}
                            else 'One folder name, without "/".')
        if name:
            await go(posixpath.join(state['path'], name))

    async def go(path: str) -> None:
        path = path.strip() or '/'
        listing.clear()
        with listing:
            ui.spinner().classes('self-center my-6')
        try:
            r = await run.io_bound(lambda: vast.remote().call('ls', timeout=45, path=path, hidden=state['hidden']))
        except Exception as e:  # noqa: BLE001
            listing.clear()
            notify_error(e)
            return
        state['path'] = r.get('path') or path
        path_input.value = state['path']
        up_btn.set_enabled(state['path'] != '/')
        free.text = f'{hub.human_size(r["free"])} free' if r.get('free') is not None else ''
        choose.set_text(f'Choose {state["path"]}')
        listing.clear()
        with listing:
            if not r.get('exists', True):
                with ui.row().classes('items-center gap-2 p-4 rz-banner info no-wrap'):
                    ui.icon('create_new_folder')
                    ui.label('This folder does not exist yet: it is created when the transfer starts.')
                return
            if r.get('error'):
                ui.label(r['error']).classes('rz-err p-4')
                return
            dirs, files = r.get('dirs', []), r.get('files', [])
            if not dirs and not files:
                ui.label('Empty folder.').classes('rulz-muted text-sm p-4')
            for d in dirs:
                target = posixpath.join(state['path'], d['name'])
                with ui.item(on_click=lambda t=target: go(t)).classes('rounded'):
                    with ui.item_section().props('avatar'):
                        ui.icon('folder').classes('rz-ic-dir')
                    with ui.item_section():
                        ui.item_label(d['name'])
            for f in files[:300]:
                with ui.item().classes('rounded opacity-60'):
                    with ui.item_section().props('avatar'):
                        ui.icon('description').classes('rz-ic-muted')
                    with ui.item_section():
                        ui.item_label(f['name'])
                    with ui.item_section().props('side'):
                        ui.label(hub.human_size(f['size'])).classes('text-xs rulz-muted')
            if len(files) > 300:
                ui.label(f'... and {len(files) - 300} more files').classes('text-xs rulz-muted p-2')

    ui.timer(0.05, lambda: go(state['path']), once=True)
    return await dialog


# --------------------------------------------------------------------------------------------------
# Send dialog
# --------------------------------------------------------------------------------------------------

async def send_dialog(sources: list[vast.Source], on_started=None) -> None:
    """Choose where on the attached instance the sources go, and how; then start the transfer."""
    if not sources:
        ui.notify('Select something first')
        return
    conn = vast.attached()
    if not conn:
        add_to_basket(sources)
        with ui.dialog() as d, ui.card().classes('rulz-card w-[480px] max-w-[95vw]'):
            ui.label('No vast.ai instance attached').classes('text-lg font-semibold')
            ui.label('Attach one on the vast.ai page first. What you picked is waiting in the transfer list there.')
            with ui.row().classes('w-full justify-end gap-2'):
                ui.button('Later', on_click=d.close).props('flat')
                ui.button('Go to the vast.ai page', icon='dns', on_click=lambda: ui.navigate.to('/vast'))
        d.open()
        return

    settings = config.get()
    state: dict = {'listed': {}, 'plan': None, 'info': None}

    with ui.dialog() as dialog, ui.card().classes('rulz-card w-[760px] max-w-[96vw] gap-3'):
        with ui.row().classes('w-full items-center gap-2 no-wrap'):
            ui.icon('cloud_upload', size='sm', color='primary')
            ui.label('Send to vast.ai').classes('text-lg font-semibold')
            ui.space()
            ui.button(icon='close', on_click=lambda: dialog.submit(None)).props('flat round dense')

        with ui.column().classes('w-full gap-1 rz-box p-3'):
            for s in sources[:8]:
                with ui.row().classes('items-center gap-2 no-wrap'):
                    ui.icon(hub.KIND_ICON[s.kind]).classes('rz-kind-icon')
                    ui.label(s.label).classes('text-sm rulz-mono ellipsis')
            if len(sources) > 8:
                ui.label(f'... and {len(sources) - 8} more').classes('text-xs rulz-muted')
            summary = ui.label('Listing files...').classes('text-sm font-medium mt-1')

        with ui.row().classes('w-full items-center gap-2 no-wrap'):
            ui.icon('dns').classes('rz-sec-icon')
            ui.label(conn.title).classes('font-medium ellipsis')
            inst_note = ui.label().classes('text-xs rulz-muted ellipsis')

        ui.label('Where on the instance').classes('text-sm font-semibold mt-1')
        with ui.row().classes('w-full items-center gap-2 no-wrap'):
            dest = ui.input(value=(settings.vast_recent_dests or ['/workspace/models'])[0]) \
                .props('outlined dense').classes('flex-1 rulz-mono')
            ui.button('Browse...', icon='folder_open', on_click=lambda: browse()).props('flat no-caps')
        free_label = ui.label().classes('text-xs rulz-muted')
        chips = ui.row().classes('w-full gap-1')

        ui.label('How').classes('text-sm font-semibold mt-1')
        layout = ui.radio(vast.LAYOUTS, value=settings.vast_layout if settings.vast_layout in vast.LAYOUTS
                          else 'keep').props('dense')
        example = ui.label().classes('text-xs rulz-muted rulz-mono')

        with ui.expansion('Only some files', icon='filter_alt').classes('w-full rz-box').props('dense'):
            with ui.row().classes('w-full gap-2 no-wrap'):
                include = ui.input('Only files matching', placeholder='*.safetensors, *.json') \
                    .props('outlined dense clearable').classes('flex-1 rulz-mono')
                exclude = ui.input('Skip files matching', placeholder='*.bin, *.pt, onnx/*') \
                    .props('outlined dense clearable').classes('flex-1 rulz-mono')
            ui.label('Patterns match the file name or its path in the repo. Files already on the instance with the '
                     'same size are skipped, so sending again only fetches what is missing.') \
                .classes('text-xs rulz-muted')

        warn = ui.row().classes('w-full items-center gap-2 rz-warn no-wrap')
        warn.set_visibility(False)
        with ui.row().classes('w-full justify-end gap-2'):
            ui.button('Add to transfer list', icon='playlist_add',
                      on_click=lambda: (add_to_basket(sources), dialog.submit(None))).props('flat no-caps')
            ui.button('Cancel', on_click=lambda: dialog.submit(None)).props('flat')
            start_btn = ui.button('Send', icon='send', on_click=lambda: start())
            start_btn.disable()

    def render_chips(info: dict | None) -> None:
        chips.clear()
        options: list[tuple[str, str]] = []
        for p in settings.vast_recent_dests[:4]:
            options.append((p, p))
        if info:
            for s in info.get('suggestions', []):
                options.append((s['label'], s['path']))
        seen = set()
        with chips:
            for label, path in options:
                if path in seen:
                    continue
                seen.add(path)
                ui.chip(label, icon='history' if label == path else 'folder',
                        on_click=lambda p=path: set_dest(p)).props('clickable outline dense').classes('text-xs')

    async def set_dest(path: str) -> None:
        dest.value = path
        if posixpath.basename(path.rstrip('/')) in COMFY_DIRS and layout.value == 'keep':
            layout.value = 'flat'  # ComfyUI only looks in its model folders and their direct subfolders
        await check_space()

    async def browse() -> None:
        picked = await pick_vast_folder(dest.value or '/workspace')
        if picked:
            await set_dest(picked)

    async def check_space() -> None:
        path = (dest.value or '').strip()
        if not path:
            free_label.text = ''
            return
        try:
            r = await run.io_bound(lambda: vast.Remote(conn).call('df', timeout=30, path=path))
        except Exception as e:  # noqa: BLE001
            free_label.text = f'Could not check the space there: {e}'
            return
        state['free'] = r.get('free')
        free_label.text = (f'{hub.human_size(r["free"])} free there' if r.get('free') is not None else '') + \
            ('' if r.get('exists') else '  (the folder is created)')
        update_warning()

    dest.on('blur', check_space)
    dest.on('keydown.enter', check_space)

    def update_warning() -> None:
        plan: vast.Plan | None = state['plan']
        msgs = []
        if plan and state.get('free') is not None and plan.total > state['free']:
            msgs.append(f'Needs {hub.human_size(plan.total)} but only {hub.human_size(state["free"])} is free there.')
        if plan and (clashes := plan.collisions()):
            msgs.append(f'{len(clashes)} file(s) would get the same name (e.g. {clashes[0]}): '
                        'keep the folders or use a folder per repo.')
        if plan is not None and not plan.items:
            msgs.append('No file matches.')
        warn.clear()
        with warn:
            ui.icon('warning')
            ui.label(' '.join(msgs)).classes('text-sm')
        warn.set_visibility(bool(msgs))
        start_btn.set_enabled(bool(plan and plan.items) and not (plan and plan.collisions()))

    def replan() -> None:
        if len(state['listed']) < len(sources):
            return
        try:
            plan = vast.build_plan(sources, layout.value, vast.split_patterns(include.value),
                                   vast.split_patterns(exclude.value), listed=state['listed'])
        except ValueError as e:
            state['plan'] = None
            summary.text = str(e)
            start_btn.disable()
            return
        state['plan'] = plan
        skipped = f', {plan.skipped_by_filter} left out by the filter' if plan.skipped_by_filter else ''
        summary.text = f'{len(plan.items)} file(s), {hub.human_size(plan.total)}{skipped}'
        if plan.items:
            sample = max(plan.items, key=lambda i: i.size)
            where = (state['info'] or {}).get('hf_home', ['HF_HOME'])[0] if layout.value == 'cache' else dest.value
            example.text = (f'e.g. {sample.repo}/{sample.path}  ->  ' +
                            (f'{where}/hub/{sample.kind}s--{sample.repo.replace("/", "--")}/...'
                             if layout.value == 'cache' else posixpath.join(where or '', sample.dest)))
        update_warning()

    def on_layout() -> None:
        settings.vast_layout = layout.value
        config.save()
        if layout.value == 'cache' and state['info'] and state['info'].get('hf_home'):
            dest.value = state['info']['hf_home'][0]
            ui.timer(0.01, check_space, once=True)
        replan()

    layout.on_value_change(on_layout)
    include.on_value_change(replan)
    exclude.on_value_change(replan)

    async def start() -> None:
        plan = state['plan']
        path = (dest.value or '').strip()
        if not plan or not path:
            return
        title = sources[0].label if len(sources) == 1 else f'{len(sources)} items'
        start_btn.disable()
        try:
            st = await run.io_bound(lambda: vast.start_transfer(plan, path, layout.value, title))
        except Exception as e:  # noqa: BLE001
            notify_error(e, prefix='Could not start: ')
            start_btn.enable()
            return
        vast.remember_destination(path)
        follow_job(st['id'], f'To vast.ai: {title}', conn)
        dialog.submit(True)
        if on_started:
            on_started()

    async def load() -> None:
        async def info() -> None:
            try:
                state['info'] = await instance_info()
            except Exception as e:  # noqa: BLE001
                inst_note.text = f'(not reachable: {e})'
                return
            i = state['info'] or {}
            ws = i.get('workspace') or i.get('home_disk') or {}
            bits = [', '.join(i.get('gpus', [])[:1])]
            if ws.get('free') is not None:
                bits.append(f'{hub.human_size(ws["free"])} free')
            if i.get('running'):
                bits.append(f'{i["running"]} transfer(s) running')
            inst_note.text = ' | '.join(b for b in bits if b)
            render_chips(i)

        async def files() -> None:
            for n, s in enumerate(sources):
                try:
                    state['listed'][n] = await run.io_bound(vast.list_source_files, s)
                except Exception as e:  # noqa: BLE001
                    summary.text = f'{s.repo_id}: {hub.friendly_error(e)}'
                    return
            replan()

        render_chips(None)
        await gather_in(chips, info(), files(), check_space())

    ui.timer(0.05, load, once=True)
    await dialog


def follow_job(job_id: str, title: str, conn: vast.Conn | None = None) -> Job:
    """A job in the Activity panel that mirrors a transfer on the instance."""
    conn = conn or vast.attached()

    def work(job: Job) -> str:
        job.set(message='Starting on the instance...')
        return vast.describe_outcome(vast.follow(job_id, job, conn))

    return run_job(title, work, icon='cloud_upload', success=lambda job: f'Sent: {job.result}')


# --------------------------------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------------------------------

@ui.page('/vast')
async def vast_page() -> None:
    settings = config.get()
    state: dict = {'instances': [], 'jobs': [], 'busy': False}

    with frame('vast'):
        ui.label('vast.ai').classes('rz-page-title')
        ui.label('Send models and whole repos from Hugging Face straight to your vast.ai instance. The instance '
                 'downloads them itself, at datacenter speed: nothing passes through this computer, and transfers '
                 'keep going when you close the app.').classes('text-sm rulz-muted -mt-2')

        # ------------------------------------------------------------------ instance
        with ui.card().classes('rulz-card w-full p-5 gap-3'):
            with ui.row().classes('w-full items-center gap-3 no-wrap'):
                ui.icon('dns').classes('rz-sec-icon')
                ui.label('Instance').classes('rz-title')
                ui.space()
                ui.button(icon='refresh', on_click=lambda: load_instances()).props('flat round') \
                    .tooltip('Refresh the list from vast.ai')
            attached_box = ui.column().classes('w-full gap-2')
            instances_box = ui.column().classes('w-full gap-2')
            with ui.expansion('Connect with an ssh command instead', icon='terminal').classes('w-full rz-box') \
                    .props('dense'):
                ui.label('Paste the command from the instance\'s "Connect" button on vast.ai (or user@host:port). '
                         'Works without an API key, and for any Linux machine you reach with ssh.') \
                    .classes('text-xs rulz-muted')
                with ui.row().classes('w-full items-center gap-2 no-wrap'):
                    ssh_input = ui.input(placeholder='ssh -p 12345 root@ssh4.vast.ai -L 8080:localhost:8080') \
                        .props('outlined dense').classes('flex-1 rulz-mono')
                    ui.button('Connect', icon='link', on_click=lambda: attach_manual())
            with ui.expansion('vast.ai API key and ssh key', icon='key').classes('w-full rz-box').props('dense'):
                key_note = ui.label().classes('text-xs rulz-muted')
                with ui.row().classes('w-full items-center gap-2 no-wrap'):
                    key_input = ui.input('vast.ai API key', value=settings.vast_api_key, password=True,
                                         password_toggle_button=True,
                                         placeholder='empty: use the vastai CLI\'s key') \
                        .props('outlined dense').classes('flex-1 rulz-mono')
                    ui.button('Save', icon='save', on_click=lambda: save_keys())
                    ui.button('Get a key', icon='open_in_new',
                              on_click=lambda: ui.navigate.to('https://cloud.vast.ai/manage-keys/', new_tab=True)) \
                        .props('flat no-caps')
                ssh_key_input = ui.input('ssh private key file', value=settings.ssh_key,
                                         placeholder='empty: ssh\'s defaults (~/.ssh/id_ed25519, id_rsa...)') \
                    .props('outlined dense').classes('w-full rulz-mono')
                ui.label('The key\'s public half must be in your vast.ai account (Keys) before the instance is '
                         'rented, or added to the instance afterwards. Keys are stored only on this computer.') \
                    .classes('text-xs rulz-muted')

        # ------------------------------------------------------------------ command prompt
        with ui.card().classes('rulz-card w-full p-5 gap-3'):
            with ui.row().classes('w-full items-center gap-3 no-wrap'):
                ui.icon('terminal').classes('rz-sec-icon')
                ui.label('Command prompt').classes('rz-title')
                ui.space()
                ui.button('Clear', icon='clear_all', on_click=lambda: clear_shell()).props('flat no-caps')
                ui.button('Copy ssh command', icon='content_copy', on_click=lambda: copy_ssh()) \
                    .props('flat no-caps').tooltip('To paste into a terminal of your own')
                ui.button('Open terminal', icon='open_in_new', on_click=lambda: open_terminal()) \
                    .props('flat no-caps').tooltip('A terminal window on this computer, logged in to the instance')
            ui.label('Runs one command at a time on the attached instance, in a bash shell with vast\'s Python venv '
                     'on PATH. The folder carries over to the next command (cd works); variables do not. Stopping '
                     'or closing the app ends the command: start long jobs with nohup, or use Open terminal (vast '
                     'keeps that session in tmux).').classes('text-xs rulz-muted')
            shell_log = ui.log(max_lines=3000).classes('w-full h-72 rulz-mono text-xs rz-box p-2')
            shell_progress = ui.label().classes('w-full rulz-mono text-xs rulz-muted ellipsis')
            shell_progress.set_visibility(False)
            with ui.row().classes('w-full items-center gap-2 no-wrap'):
                shell_cwd = ui.label().classes('rulz-mono text-sm whitespace-nowrap')
                shell_input = ui.input(placeholder='nvidia-smi, ls -la, df -h, pip list | grep torch...') \
                    .props('outlined dense autocomplete=off spellcheck=false').classes('flex-1 rulz-mono')
                shell_run_btn = ui.button('Run', icon='play_arrow', on_click=lambda: shell_enter())
                shell_stop_btn = ui.button('Stop', icon='stop', color='negative', on_click=lambda: stop_shell()) \
                    .props('outline')
                shell_stop_btn.set_visibility(False)
            with ui.row().classes('w-full items-center gap-1'):
                for cmd in ('nvidia-smi', 'df -h /workspace', 'ls -la', 'free -h', 'pip list'):
                    ui.button(cmd, on_click=lambda c=cmd: run_shell(c)).props('flat dense no-caps size=sm') \
                        .classes('rulz-mono')
            shell_input.on('keydown.enter', lambda: shell_enter())
            shell_input.on('keydown', lambda e: shell_history(-1 if e.args == 'ArrowUp' else 1),
                           js_handler='(e) => { if (e.key === "ArrowUp" || e.key === "ArrowDown") '
                                      '{ e.preventDefault(); emit(e.key); } }')

        # ------------------------------------------------------------------ pick
        with ui.card().classes('rulz-card w-full p-5 gap-3'):
            with ui.row().classes('items-center gap-3 no-wrap'):
                ui.icon('playlist_add').classes('rz-sec-icon')
                ui.label('Pick what to send').classes('rz-title')
            with ui.row().classes('w-full gap-2 items-start text-sm'):
                ui.icon('lightbulb').classes('rz-ic-muted')
                ui.markdown('Tick repos on **My Hub** (yours, an organization\'s, or anyone\'s: type a username '
                            'there), or files and folders in any repo, and press **To vast.ai**. Or paste links '
                            'below, or search the Hub.').classes('flex-1')
            with ui.row().classes('w-full gap-2 items-start no-wrap'):
                paste = ui.textarea(placeholder='One per line: user/model, a huggingface.co link (repo, folder or '
                                                'file), datasets/user/name, hf://...') \
                    .props('outlined dense autogrow').classes('flex-1 rulz-mono')
                ui.button('Add', icon='add', on_click=lambda: add_pasted())
            with ui.row().classes('w-full gap-2 items-center no-wrap'):
                find = ui.input(placeholder='Search Hugging Face (e.g. flux, wan 2.2, qwen image)...') \
                    .props('outlined dense clearable').classes('flex-1')
                with find.add_slot('prepend'):
                    ui.icon('search')
                find_kind = ui.toggle({'model': 'Models', 'dataset': 'Datasets'}, value='model') \
                    .props('no-caps rounded unelevated toggle-color=primary dense')
                ui.button('Search', on_click=lambda: search()).props('flat no-caps')
            find.on('keydown.enter', lambda: search())
            results = ui.column().classes('w-full gap-0')

        # ------------------------------------------------------------------ basket
        with ui.card().classes('rulz-card w-full p-5 gap-3'):
            with ui.row().classes('w-full items-center gap-3 no-wrap'):
                ui.icon('shopping_basket').classes('rz-sec-icon')
                ui.label('Transfer list').classes('rz-title')
                ui.space()
                ui.button('Clear', icon='clear_all', on_click=lambda: (BASKET.clear(), render_basket())) \
                    .props('flat no-caps')
                send_btn = ui.button('Send to vast.ai...', icon='send', on_click=lambda: send_basket())
            basket_box = ui.column().classes('w-full gap-1')

        # ------------------------------------------------------------------ remote jobs
        with ui.card().classes('rulz-card w-full p-5 gap-3'):
            with ui.row().classes('w-full items-center gap-3 no-wrap'):
                ui.icon('downloading').classes('rz-sec-icon')
                ui.label('Transfers on the instance').classes('rz-title')
                ui.space()
                ui.button(icon='refresh', on_click=lambda: load_jobs()).props('flat round').tooltip('Refresh')
            jobs_box = ui.column().classes('w-full gap-2')

    # ======================================================================== instance
    def render_attached(info: dict | None = None, error: str = '') -> None:
        conn = vast.attached()
        render_shell()
        attached_box.clear()
        with attached_box:
            if not conn:
                with ui.row().classes('w-full items-center gap-2 rz-banner info no-wrap'):
                    ui.icon('link_off')
                    ui.label('No instance attached. Pick a running one below.')
                return
            with ui.row().classes('w-full items-center gap-3 rz-box p-3'):
                ui.icon('check_circle' if not error else 'error', color='positive' if not error else 'negative',
                        size='sm')
                with ui.column().classes('gap-0 flex-1 min-w-0'):
                    ui.label(f'Attached: {conn.title}').classes('font-semibold')
                    bits = [f'{conn.user}@{conn.host}:{conn.port} ({conn.route})']
                    if info:
                        bits += info.get('gpus', [])[:2]
                        ws = info.get('workspace') or info.get('home_disk') or {}
                        if ws.get('free') is not None:
                            bits.append(f'{hub.human_size(ws["free"])} free of {hub.human_size(ws["total"])}')
                    ui.label(' | '.join(bits)).classes('text-xs rulz-muted')
                    if error:
                        ui.label(error).classes('text-xs rz-err whitespace-pre-line')
                ui.button('Test', icon='network_check', on_click=lambda: test_attached()).props('flat no-caps')
                ui.button('Browse files', icon='folder_open', on_click=lambda: pick_vast_folder(
                    '/workspace', 'Files on the instance')).props('flat no-caps')
                ui.button('Detach', icon='link_off', on_click=lambda: detach()).props('flat no-caps')

    async def test_attached() -> None:
        if not vast.attached():
            return
        try:
            info = await instance_info(force=True)
            render_attached(info)
            notify_ok('The instance answers')
        except Exception as e:  # noqa: BLE001
            render_attached(error=hub.friendly_error(e) if not isinstance(e, vast.VastError) else str(e))
        await load_jobs()

    def detach() -> None:
        vast.save_attached(None)
        _info.update(conn=None, info=None)
        render_attached()
        render_instances()
        render_jobs()

    def render_instances(error: str = '') -> None:
        instances_box.clear()
        with instances_box:
            if error:
                with ui.row().classes('w-full items-center gap-2 rz-banner info no-wrap'):
                    ui.icon('info')
                    ui.label(error).classes('text-sm')
                return
            if not state['instances']:
                ui.label('No instances on your vast.ai account. Rent one on cloud.vast.ai, then refresh.') \
                    .classes('text-sm rulz-muted')
                return
            conn = vast.attached()
            for inst in state['instances']:
                with ui.row().classes('w-full items-center gap-3 rz-box px-3 py-2 no-wrap'):
                    ui.icon('memory', color='primary' if inst.running else 'grey')
                    with ui.column().classes('gap-0 flex-1 min-w-0'):
                        ui.label(inst.title).classes('font-medium ellipsis')
                        bits = [inst.status or '?']
                        if inst.dollars_per_hour:
                            bits.append(f'${inst.dollars_per_hour:.3f}/h')
                        if inst.disk_gb:
                            bits.append(f'{inst.disk_gb:.0f} GB disk')
                        if inst.image:
                            bits.append(inst.image)
                        ui.label(' | '.join(bits)).classes('text-xs rulz-muted ellipsis')
                    if conn and conn.instance_id == inst.id:
                        ui.badge('attached', color='positive')
                    else:
                        b = ui.button('Attach', icon='link', on_click=lambda i=inst: attach(i)).props('no-caps')
                        b.set_enabled(inst.running)

    async def load_instances() -> None:
        key_note.text = (f'Using the key from {vast.key_source()}.' if vast.key_source()
                         else 'No key yet: paste one, or run "vastai set api-key <key>" once.')
        if not vast.api_key():
            state['instances'] = []
            render_instances('Add your vast.ai API key below to list your instances, or connect with an ssh '
                             'command.')
            return
        instances_box.clear()
        with instances_box:
            ui.spinner().classes('self-center')
        try:
            state['instances'] = await run.io_bound(vast.list_instances)
        except Exception as e:  # noqa: BLE001
            render_instances(str(e))
            return
        render_instances()

    async def attach(inst: vast.Instance) -> None:
        n = ui.notification(f'Connecting to {inst.title}...', type='ongoing', spinner=True, timeout=None)
        try:
            conn, info = await run.io_bound(vast.attach_instance, inst)
        except Exception as e:  # noqa: BLE001
            n.dismiss()
            notify_error(e)
            return
        n.dismiss()
        _info.update(conn=(conn.host, conn.port), info=info, at=time.time())
        notify_ok(f'Attached {inst.title} ({conn.route} route)')
        render_attached(info)
        render_instances()
        await load_jobs()

    async def attach_manual() -> None:
        try:
            conn = vast.parse_ssh_command(ssh_input.value or '')
        except ValueError as e:
            notify_error(str(e))
            return
        n = ui.notification(f'Connecting to {conn.title}...', type='ongoing', spinner=True, timeout=None)
        try:
            conn, info = await run.io_bound(vast.attach_manual, conn)
        except Exception as e:  # noqa: BLE001
            n.dismiss()
            notify_error(e)
            return
        n.dismiss()
        _info.update(conn=(conn.host, conn.port), info=info, at=time.time())
        notify_ok(f'Attached {conn.title}')
        render_attached(info)
        render_instances()
        await load_jobs()

    async def save_keys() -> None:
        settings.vast_api_key = (key_input.value or '').strip()
        settings.ssh_key = (ssh_key_input.value or '').strip()
        config.save()
        notify_ok('Saved')
        await load_instances()

    # ======================================================================== picking
    async def add_pasted() -> None:
        lines = [x.strip() for x in (paste.value or '').splitlines() if x.strip()]
        if not lines:
            return
        added, bad = [], []
        for line in lines:
            try:
                added.append(await run.io_bound(vast.source_from_text, line))
            except Exception as e:  # noqa: BLE001
                bad.append(f'{line}: {e}')
        if added:
            add_to_basket(added)
            render_basket()
        paste.value = '\n'.join(b.split(': ', 1)[0] for b in bad)
        if bad:
            notify_error('Could not add:\n' + '\n'.join(bad[:5]))

    async def search() -> None:
        q = (find.value or '').strip()
        results.clear()
        if not q:
            return
        with results:
            ui.spinner().classes('self-center')
        kind = find_kind.value

        def query() -> list:
            a = hub.api()
            if kind == 'dataset':
                return list(a.list_datasets(search=q, sort='downloads', limit=25))
            return list(a.list_models(search=q, sort='downloads', limit=25))
        try:
            found = await run.io_bound(query)
        except Exception as e:  # noqa: BLE001
            results.clear()
            notify_error(e)
            return
        results.clear()
        with results:
            if not found:
                ui.label('Nothing found.').classes('text-sm rulz-muted')
            for r in found:
                with ui.row().classes('w-full items-center gap-2 no-wrap px-2 py-1 rz-hover rounded'):
                    ui.icon(hub.KIND_ICON[kind]).classes('rz-kind-icon')
                    ui.label(r.id).classes('font-medium ellipsis flex-1 cursor-pointer') \
                        .on('click', lambda r=r: ui.navigate.to(browse_url(kind, r.id)))
                    bits = []
                    if getattr(r, 'downloads', None) is not None:
                        bits.append(f'{r.downloads:,} downloads')
                    if getattr(r, 'likes', None):
                        bits.append(f'{r.likes} likes')
                    ui.label(' / '.join(bits)).classes('text-xs rulz-muted whitespace-nowrap')
                    ui.button(icon='folder_open', on_click=lambda r=r: ui.navigate.to(browse_url(kind, r.id))) \
                        .props('flat round dense size=sm').tooltip('Browse files (pick single files there)')
                    ui.button(icon='add', on_click=lambda r=r: (add_to_basket([vast.Source(kind, r.id)]),
                                                                 render_basket())) \
                        .props('flat round dense size=sm').tooltip('Add the whole repo to the transfer list')

    # ======================================================================== basket
    def render_basket() -> None:
        basket_box.clear()
        send_btn.set_enabled(bool(BASKET))
        with basket_box:
            if not BASKET:
                ui.label('Nothing yet.').classes('text-sm rulz-muted')
                return
            for n, s in enumerate(list(BASKET)):
                with ui.row().classes('w-full items-center gap-2 no-wrap rz-box px-3 py-1'):
                    ui.icon(hub.KIND_ICON[s.kind]).classes('rz-kind-icon')
                    ui.label(s.label).classes('rulz-mono text-sm ellipsis flex-1')
                    if s.revision:
                        ui.badge(s.revision).classes('rz-badge')
                    ui.button(icon='folder_open', on_click=lambda s=s: ui.navigate.to(
                        browse_url(s.kind, s.repo_id, rev=s.revision))).props('flat round dense size=sm')
                    ui.button(icon='close', on_click=lambda n=n: (BASKET.pop(n), render_basket())) \
                        .props('flat round dense size=sm').tooltip('Remove')

    async def send_basket() -> None:
        def started() -> None:
            BASKET.clear()
            render_basket()
            ui.timer(1.5, load_jobs, once=True)
        await send_dialog(list(BASKET), on_started=started)

    # ======================================================================== remote jobs
    def render_jobs(error: str = '') -> None:
        jobs_box.clear()
        with jobs_box:
            if not vast.attached():
                ui.label('Attach an instance to see its transfers.').classes('text-sm rulz-muted')
                return
            if error:
                ui.label(error).classes('text-sm rz-err')
                return
            if not state['jobs']:
                ui.label('No transfers on this instance yet.').classes('text-sm rulz-muted')
                return
            for st in state['jobs']:
                job_card(st)

    def job_card(st: dict) -> None:
        s = st.get('state') or '?'
        icon, color = STATE_LOOK.get(s, ('help', 'grey'))
        total = st.get('bytes_total') or 0
        done = (st.get('bytes_done') or 0) + (st.get('partial') or 0)
        with ui.column().classes('w-full gap-1 rz-box p-3'):
            with ui.row().classes('w-full items-center gap-2 no-wrap'):
                if s in {'running', 'installing', 'starting', 'queued'}:
                    ui.spinner(size='sm', color='primary')
                else:
                    ui.icon(icon, color=color)
                ui.label(st.get('title') or st.get('id')).classes('font-medium ellipsis flex-1')
                ui.label(s).classes('text-xs uppercase rulz-muted')
                if s in {'running', 'installing', 'starting', 'queued'}:
                    ui.button('Follow', icon='visibility', on_click=lambda: follow_job(
                        st['id'], f'To vast.ai: {st.get("title") or st["id"]}')).props('flat dense no-caps size=sm') \
                        .tooltip('Show its progress in the Activity panel')
                    ui.button('Cancel', icon='block', on_click=lambda: cancel(st)) \
                        .props('flat dense no-caps size=sm color=negative')
                else:
                    if s in {'error', 'stopped', 'cancelled'}:
                        ui.button('Retry', icon='replay', on_click=lambda: retry(st)) \
                            .props('flat dense no-caps size=sm').tooltip('Resume: files already there are skipped')
                    ui.button(icon='delete_outline', on_click=lambda: forget(st)) \
                        .props('flat round dense size=sm').tooltip('Remove from this list (files stay)')
            if total and s not in {'done'}:
                ui.linear_progress(value=min(1.0, done / total), show_value=False).props('rounded size=6px')
            bits = [f'{st.get("files_done", 0)}/{st.get("files_total", 0)} files',
                    f'{hub.human_size(done)} of {hub.human_size(total)}' if total else '']
            if s == 'running' and (st.get('speed') or 0) > 1024:
                bits.append(f'{hub.human_size(st["speed"])}/s')
            if st.get('skipped'):
                bits.append(f'{st["skipped"]} already there')
            if st.get('started') and st.get('finished'):
                bits.append(fmt_duration(st['finished'] - st['started']))
            ui.label(' | '.join(b for b in bits if b) + f'  ->  {st.get("dest")}').classes('text-xs rulz-muted '
                                                                                         'rulz-mono ellipsis')
            if s == 'installing' and st.get('message'):
                ui.label(st['message']).classes('text-xs rulz-muted')
            if st.get('error'):
                ui.label(st['error']).classes('text-xs rz-err whitespace-pre-line')
            for f in (st.get('failed') or [])[:3]:
                ui.label(f'{f["file"]}: {f["error"]}').classes('text-xs rz-err rulz-mono ellipsis')

    async def load_jobs() -> None:
        if not vast.attached():
            render_jobs()
            return
        if state['busy']:
            return
        state['busy'] = True
        try:
            r = await run.io_bound(lambda: vast.remote().call('jobs', timeout=45))
            state['jobs'] = r.get('jobs', [])
            render_jobs()
        except Exception as e:  # noqa: BLE001
            render_jobs(str(e))
        finally:
            state['busy'] = False

    async def cancel(st: dict) -> None:
        if not await confirm('Cancel this transfer?', st.get('title') or '', ok_label='Cancel transfer',
                             ok_color='negative', warning='Files already finished stay; Retry resumes the rest.'):
            return
        try:
            await run.io_bound(lambda: vast.remote().call('cancel', id=st['id']))
        except Exception as e:  # noqa: BLE001
            notify_error(e)
        await load_jobs()

    async def retry(st: dict) -> None:
        try:
            await run.io_bound(vast.retry_transfer, st['id'])
        except Exception as e:  # noqa: BLE001
            notify_error(e)
            return
        follow_job(st['id'], f'To vast.ai: {st.get("title") or st["id"]}')
        await load_jobs()

    async def forget(st: dict) -> None:
        try:
            await run.io_bound(lambda: vast.remote().call('forget', id=st['id']))
        except Exception as e:  # noqa: BLE001
            notify_error(e)
        await load_jobs()

    # ======================================================================== command prompt
    shell: dict = {'run': None, 'queue': queue.SimpleQueue(), 'hpos': len(SHELL['history']), 'draft': ''}

    def shell_folder() -> str:
        conn = vast.attached()
        return SHELL['cwd'].get(conn.key, '/workspace') if conn else '/workspace'

    def render_shell() -> None:
        conn = vast.attached()
        running = shell['run'] is not None
        shell_cwd.text = f'{shell_folder()} $' if conn else 'not attached'
        shell_input.set_enabled(bool(conn))
        shell_run_btn.set_visibility(not running)
        shell_run_btn.set_enabled(bool(conn))
        shell_stop_btn.set_visibility(running)
        shell_input.props['placeholder'] = ('Type an answer for the running command (Enter sends it)' if running
                                            else 'nvidia-smi, ls -la, df -h, pip list | grep torch...')
        shell_input.update()

    def shell_enter() -> None:
        text = shell_input.value or ''
        if shell['run'] is not None:
            shell['run'].send(text)
            shell_input.value = ''
            return
        if text.strip():
            run_shell(text)

    def shell_history(step: int) -> None:
        hist = SHELL['history']
        if not hist or shell['run'] is not None:
            return
        if shell['hpos'] == len(hist):
            shell['draft'] = shell_input.value or ''
        shell['hpos'] = max(0, min(len(hist), shell['hpos'] + step))
        shell_input.value = hist[shell['hpos']] if shell['hpos'] < len(hist) else shell['draft']

    def run_shell(command: str) -> None:
        command = command.strip()
        if not command or shell['run'] is not None:
            return
        conn = vast.attached()
        if not conn:
            ui.notify('Attach an instance first', type='warning')
            return
        if command in {'clear', 'cls'}:
            clear_shell()
            shell_input.value = ''
            return
        hist = SHELL['history']
        if not hist or hist[-1] != command:
            hist.append(command)
            del hist[:-200]
        shell['hpos'] = len(hist)
        shell_input.value = ''
        shell_log.push(f'{shell_folder()} $ {command}')
        try:
            proc = vast.ShellRun(conn, command, shell_folder())
        except vast.VastError as e:
            shell_log.push(str(e))
            return
        shell['run'] = proc
        q = shell['queue']

        def pump() -> None:
            try:
                for event in proc.events():
                    q.put(event)
            except Exception as e:  # noqa: BLE001
                proc.error = str(e)
                q.put(('done', ''))

        threading.Thread(target=pump, daemon=True, name='rulz-shell').start()
        render_shell()

    def drain_shell() -> None:
        q = shell['queue']
        lines: list[str] = []
        progress = None
        finished = False
        while not q.empty():
            kind, text = q.get()
            if kind == 'line':
                lines.append(text)
                progress = ''
            elif kind == 'cr':
                progress = text
            elif kind == 'done':
                finished = True
        if lines:
            shell_log.push('\n'.join(lines))
        if progress is not None:
            shell_progress.text = progress
            shell_progress.set_visibility(bool(progress.strip()))
        if finished and (proc := shell['run']) is not None:
            shell['run'] = None
            shell_progress.set_visibility(False)
            conn = vast.attached()
            if conn:
                SHELL['cwd'][conn.key] = proc.cwd
            if proc.error:
                shell_log.push(f'[{proc.error}]')
            elif proc.stopped:
                shell_log.push('[stopped]')
            elif proc.code:
                shell_log.push(f'[exit code {proc.code}]')
            render_shell()
            shell_input.run_method('focus')

    def stop_shell() -> None:
        if (proc := shell['run']) is not None:
            threading.Thread(target=proc.stop, daemon=True).start()

    def clear_shell() -> None:
        shell_log.clear()
        shell_progress.set_visibility(False)

    def copy_ssh() -> None:
        conn = vast.attached()
        if not conn:
            ui.notify('Attach an instance first', type='warning')
            return
        ui.clipboard.write(vast.ssh_login_text(conn))
        ui.notify('ssh command copied', type='positive')

    async def open_terminal() -> None:
        try:
            what = await run.io_bound(vast.open_terminal)
            notify_ok(f'Opened {what}')
        except Exception as e:  # noqa: BLE001
            notify_error(e)

    async def auto_refresh() -> None:
        if any(j.get('state') in {'running', 'installing', 'starting', 'queued'} for j in state['jobs']):
            await load_jobs()

    async def init() -> None:
        render_attached()
        render_shell()
        render_basket()
        render_jobs()
        await gather_in(attached_box, load_instances(),
                        test_attached() if vast.attached() else asyncio.sleep(0))

    ui.timer(0.01, init, once=True)
    ui.timer(5.0, auto_refresh)
    ui.timer(0.15, drain_shell)
