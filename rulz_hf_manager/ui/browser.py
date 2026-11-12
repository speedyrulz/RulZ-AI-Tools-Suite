"""File browser for a single repo (model / dataset / space) or bucket."""

from __future__ import annotations

import base64
import json
import posixpath
import shutil
import tempfile
from pathlib import Path
from urllib.parse import unquote

from nicegui import app, events, run, ui

from .. import config, hub, vast
from ..jobs import Job
from .common import (SIZE_FORMAT_JS, browse_url, commit_fields, confirm, fmt_ago, fmt_date, frame, notify_error,
                     notify_ok, prompt, run_job)
from .pickers import pick_local, pick_remote_folder
from .repo_actions import delete_dialog, duplicate_dialog, rename_dialog, visibility_dialog
from .vast_page import add_to_basket, send_dialog

TEXT_EXT = {'.md', '.txt', '.json', '.jsonl', '.yaml', '.yml', '.toml', '.py', '.js', '.ts', '.tsx', '.jsx', '.sh',
            '.cfg', '.ini', '.csv', '.tsv', '.html', '.css', '.xml', '.sql', '.gitattributes', '.gitignore',
            '.dockerfile', '.env', '.log', '.rst', '.jinja', '.j2', '.ipynb', '.tex', '.r', '.rs', '.go', '.java',
            '.c', '.h', '.cpp', '.hpp', '.license', '.conf', '.properties'}
TEXT_NAMES = {'readme', 'license', 'dockerfile', 'makefile', 'requirements.txt', '.gitattributes', '.gitignore',
              'notice', 'citation.cff', 'tokenizer.model.txt'}
IMAGE_EXT = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.bmp', '.ico'}
LANGUAGE = {'.md': 'Markdown', '.json': 'JSON', '.jsonl': 'JSON', '.ipynb': 'JSON', '.yaml': 'YAML', '.yml': 'YAML',
            '.toml': 'TOML', '.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript', '.tsx': 'TSX', '.jsx': 'JSX',
            '.sh': 'Shell', '.html': 'HTML', '.css': 'CSS', '.xml': 'XML', '.sql': 'SQL', '.dockerfile': 'Dockerfile'}
WEIGHT_EXT = {'.safetensors', '.bin', '.pt', '.pth', '.ckpt', '.gguf', '.onnx', '.h5', '.msgpack', '.tflite',
              '.pb', '.mlmodel', '.npz', '.npy', '.model', '.ggml', '.engine', '.pkl', '.joblib', '.keras'}
TABLE_EXT = {'.parquet', '.csv', '.tsv', '.arrow', '.jsonl', '.feather', '.avro', '.orc', '.xlsx', '.lance'}
PREVIEW_LIMIT = 5 * 1024 * 1024

_temp_downloads: list[Path] = []


def _cleanup_temp_downloads() -> None:
    for p in _temp_downloads:
        hub.cleanup_temp(p)


app.on_shutdown(_cleanup_temp_downloads)


def _suffix(name: str) -> str:
    name = name.lower()
    if name.startswith('.') and name.count('.') == 1:
        return name
    return '.' + name.rsplit('.', 1)[-1] if '.' in name else ''


def file_icon(name: str, is_dir: bool) -> str:
    if is_dir:
        return 'folder'
    s = _suffix(name)
    if s in WEIGHT_EXT:
        return 'memory'
    if s in {'.json', '.yaml', '.yml', '.toml', '.cfg', '.ini'}:
        return 'data_object'
    if s in {'.md', '.txt', '.rst'} or name.lower().startswith('readme'):
        return 'article'
    if s in {'.py', '.js', '.ts', '.sh', '.ipynb', '.rs', '.go', '.c', '.cpp', '.java'}:
        return 'code'
    if s in IMAGE_EXT:
        return 'image'
    if s in {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}:
        return 'audio_file'
    if s in {'.mp4', '.mov', '.webm', '.mkv', '.avi'}:
        return 'movie'
    if s in TABLE_EXT:
        return 'table_view'
    if s in {'.zip', '.tar', '.gz', '.tgz', '.7z', '.xz', '.zst'}:
        return 'folder_zip'
    if s in {'.gitattributes', '.gitignore'}:
        return 'settings'
    if s == '.pdf':
        return 'picture_as_pdf'
    return 'draft'


def is_text_file(name: str) -> bool:
    lower = name.lower()
    return _suffix(lower) in TEXT_EXT or lower in TEXT_NAMES or lower.startswith('readme')


@ui.page('/browse/{kind}/{namespace}/{name}')
async def browse_page(kind: str, namespace: str, name: str, path: str = '', rev: str = '') -> None:
    if kind not in hub.KINDS:
        ui.navigate.to('/')
        return
    settings = config.get()
    repo_id = f'{unquote(namespace)}/{unquote(name)}'
    fs = hub.open_fs(kind, repo_id, rev or None)
    is_bucket = fs.is_bucket
    state: dict = {'folder': path.strip('/'), 'entries': {}, 'virtual': set(), 'summary': {}}
    try:
        settings.add_recent(kind, repo_id)
    except OSError:
        pass

    def current_url(folder: str | None = None) -> str:
        return browse_url(kind, repo_id, state['folder'] if folder is None else folder, fs.revision)

    with frame('browse'):
        # ---------------------------------------------------------------- header card
        with ui.card().classes('rulz-card w-full p-4 gap-2'):
            # Wrapping row: on narrow windows the action buttons drop below the title instead of overlapping it.
            with ui.row().classes('w-full items-center gap-3'):
                with ui.row().classes('items-center gap-3 no-wrap min-w-0 flex-1').style('flex-basis: 320px'):
                    ui.icon(hub.KIND_ICON[kind], size='28px').classes('rz-kind-tile')
                    with ui.column().classes('gap-0 min-w-0 flex-1'):
                        ui.label(hub.KIND_LABEL[kind]).classes('text-xs uppercase rulz-muted tracking-wider')
                        with ui.row().classes('items-center gap-1 no-wrap min-w-0 w-full'):
                            ui.label(repo_id).classes('text-xl font-semibold ellipsis min-w-0').tooltip(repo_id)
                            ui.button(icon='content_copy', on_click=lambda: (ui.clipboard.write(repo_id),
                                                                               ui.notify('Copied id'))) \
                                .props('flat round dense size=sm').tooltip('Copy id')
                        badges = ui.row().classes('items-center gap-1')
                with ui.row().classes('items-center gap-1 no-wrap ml-auto'):
                    if not is_bucket:
                        branch_select = ui.select([fs.revision or 'main'], value=fs.revision or 'main',
                                                  label='Branch').props('outlined dense options-dense').classes('w-40')
                    fav = ui.button(icon='star' if settings.is_favorite(kind, repo_id) else 'star_outline',
                                    color=None).props('flat round').classes('rz-kind-icon').tooltip('Favorite')
                    ui.button('Open on HF', icon='open_in_new',
                              on_click=lambda: ui.navigate.to(hub.web_url(kind, repo_id, revision=fs.revision),
                                                              new_tab=True)).props('flat no-caps') \
                        .classes('whitespace-nowrap')
                    manage_btn = ui.button(icon='settings').props('flat round').tooltip('Manage')
                with manage_btn:
                    with ui.menu().props('auto-close'):
                        ui.menu_item('Rename / transfer...', on_click=lambda: rename_dialog(
                            kind, repo_id, on_done=lambda new: ui.navigate.to(browse_url(kind, new))))
                        ui.menu_item('Change visibility (public / private)...', on_click=lambda: visibility_dialog(
                            kind, repo_id, bool(state['summary'].get('private')), on_done=lambda _: load_summary()))
                        if not is_bucket:
                            ui.menu_item('Create branch...', on_click=lambda: new_branch())
                            ui.menu_item('Duplicate...', on_click=lambda: duplicate_dialog(kind, repo_id))
                        ui.separator()
                        ui.menu_item('Delete...', on_click=lambda: delete_dialog(
                            kind, repo_id, on_done=lambda: ui.navigate.to('/'))).classes('text-negative')

        def toggle_fav() -> None:
            added = settings.toggle_favorite(kind, repo_id)
            fav.props(f'icon={"star" if added else "star_outline"}')
            ui.notify('Added to favorites' if added else 'Removed from favorites')
        fav.on_click(toggle_fav)

        # ---------------------------------------------------------------- tabs
        tabs = None
        if not is_bucket:
            with ui.tabs().props('dense align=left no-caps inline-label').classes('w-full rz-tabs') as tabs:
                ui.tab('files', 'Files', icon='folder')
                ui.tab('history', 'History', icon='history')
        panels = ui.tab_panels(tabs, value='files', animated=False).classes('w-full bg-transparent')
        with panels:
            with ui.tab_panel('files').classes('p-0 gap-3'):
                # ---------------------------------------------------------- toolbar
                with ui.card().classes('rulz-card w-full p-2 gap-2'):
                    with ui.row().classes('w-full items-center gap-1'):
                        with ui.button('Upload', icon='upload').props('unelevated no-caps'):
                            with ui.menu().props('auto-close'):
                                with ui.menu_item(on_click=lambda: upload_from_computer()):
                                    with ui.row().classes('items-center gap-3 no-wrap'):
                                        ui.icon('computer')
                                        with ui.column().classes('gap-0'):
                                            ui.label('Files or folders from this computer')
                                            ui.label('Best for big files: read straight from disk').classes(
                                                'text-xs rulz-muted')
                                with ui.menu_item(on_click=lambda: upload_from_browser()):
                                    with ui.row().classes('items-center gap-3 no-wrap'):
                                        ui.icon('file_upload')
                                        with ui.column().classes('gap-0'):
                                            ui.label('Drag & drop files')
                                            ui.label('Through the browser').classes('text-xs rulz-muted')
                                with ui.menu_item(on_click=lambda: sync_dialog()):
                                    with ui.row().classes('items-center gap-3 no-wrap'):
                                        ui.icon('sync')
                                        with ui.column().classes('gap-0'):
                                            ui.label('Sync a folder...' if is_bucket else 'Push a whole folder...')
                                            ui.label('Only changed files are transferred').classes(
                                                'text-xs rulz-muted')
                        ui.button('New folder', icon='create_new_folder', on_click=lambda: new_folder()) \
                            .props('flat no-caps')
                        ui.separator().props('vertical').classes('mx-1')
                        with ui.button('Download', icon='download').props('flat no-caps') as dl_btn:
                            with ui.menu().props('auto-close'):
                                ui.menu_item('To my download folder', on_click=lambda: download_selected(False))
                                ui.menu_item('Through the browser (single file / zip)',
                                             on_click=lambda: download_selected(True))
                                if is_bucket:
                                    ui.menu_item('Sync bucket folder to this computer...',
                                                 on_click=lambda: sync_dialog('down'))
                        with ui.button('To vast.ai', icon='cloud_upload').props('flat no-caps') as vast_btn:
                            with ui.menu().props('auto-close'):
                                ui.menu_item('Send to my vast.ai instance...',
                                             on_click=lambda: send_dialog([vast_source()]))
                                ui.menu_item('Add to the vast.ai transfer list',
                                             on_click=lambda: add_to_basket([vast_source()]))
                        vast_btn.tooltip('Straight from Hugging Face to your vast.ai instance: the ticked files '
                                         'and folders, or everything here')
                        rename_btn = ui.button('Rename', icon='drive_file_rename_outline',
                                               on_click=lambda: rename_selected()).props('flat no-caps')
                        move_btn = ui.button('Move', icon='drive_file_move', on_click=lambda: move_selected(False)) \
                            .props('flat no-caps')
                        copy_btn = ui.button('Copy', icon='file_copy', on_click=lambda: move_selected(True)) \
                            .props('flat no-caps')
                        copy_to_btn = ui.button('Copy to...', icon='move_up', on_click=lambda: copy_to_dialog()) \
                            .props('flat no-caps').tooltip('Copy into another repo or bucket (server-side)')
                        delete_btn = ui.button('Delete', icon='delete_outline', on_click=lambda: delete_selected()) \
                            .props('flat no-caps color=negative')
                        ui.space()
                        search = ui.input(placeholder='Filter...').props('dense outlined clearable').classes('w-48')
                        with search.add_slot('prepend'):
                            ui.icon('filter_list')
                        ui.button(icon='refresh', on_click=lambda: refresh()).props('flat round').tooltip('Refresh')
                    with ui.row().classes('w-full items-center gap-0 px-1 rulz-crumb no-wrap') as crumbs:
                        pass

                virtual_note = ui.row().classes('w-full items-center gap-2 no-wrap rz-banner info')
                with virtual_note:
                    ui.icon('info')
                    ui.label('This folder is new. It will appear on the Hub as soon as you upload files into it.')
                virtual_note.set_visibility(False)

                # ---------------------------------------------------------- table
                columns = [
                    {'name': 'name', 'label': 'Name', 'field': 'name', 'align': 'left', 'sortable': True,
                     ':sort': '(a, b, rowA, rowB) => (rowB.is_dir - rowA.is_dir) || a.localeCompare(b, undefined, '
                              '{numeric: true, sensitivity: "base"})'},
                    {'name': 'size', 'label': 'Size', 'field': 'size', 'align': 'right', 'sortable': True,
                     ':format': SIZE_FORMAT_JS},
                    {'name': 'modified', 'label': 'Modified', 'field': 'modified', 'align': 'left',
                     'sortable': True},
                    {'name': 'actions', 'label': '', 'field': 'path', 'align': 'right'},
                ]
                table = ui.table(columns=columns, rows=[], row_key='path', selection='multiple',
                                 pagination={'rowsPerPage': settings.page_size, 'sortBy': 'name'}) \
                    .props('flat bordered binary-state-sort :rows-per-page-options="[50,100,250,0]" '
                           'no-data-label="This folder is empty." loading-label="Loading..." '
                           'no-results-label="No file matches the filter."') \
                    .classes('w-full rulz-table')
                table.bind_filter_from(search, 'value')
                table.add_slot('body-cell-name', r'''
                    <q-td :props="props">
                      <div class="row items-center no-wrap q-gutter-x-sm cursor-pointer rulz-name"
                           @click="$parent.$emit('open_entry', props.row)">
                        <q-icon :name="props.row.icon" size="20px"
                                :class="props.row.is_dir ? 'rz-ic-dir' : (props.row.icon === 'memory' ? 'rz-ic-weight' : 'rz-ic-file')"/>
                        <span :class="props.row.is_dir ? 'text-weight-medium' : ''">{{ props.row.name }}</span>
                        <q-badge v-if="props.row.storage" class="rz-badge" :label="props.row.storage"/>
                        <q-badge v-if="props.row.virtual" class="rz-badge rz-new" label="new"/>
                      </div>
                    </q-td>''')
                table.add_slot('body-cell-modified', r'''
                    <q-td :props="props"><span :title="props.row.modified">{{ props.row.modified_ago }}</span></q-td>''')
                table.add_slot('body-cell-actions', r'''
                    <q-td :props="props" auto-width>
                      <q-btn v-if="!props.row.is_dir && props.row.previewable" flat dense round size="sm" icon="visibility"
                             @click.stop="$parent.$emit('row_action', {action: 'preview', row: props.row})">
                        <q-tooltip>Preview / edit</q-tooltip></q-btn>
                      <q-btn flat dense round size="sm" icon="drive_file_rename_outline"
                             @click.stop="$parent.$emit('row_action', {action: 'rename', row: props.row})">
                        <q-tooltip>Rename</q-tooltip></q-btn>
                      <q-btn flat dense round size="sm" icon="download"
                             @click.stop="$parent.$emit('row_action', {action: 'download', row: props.row})">
                        <q-tooltip>Download to my download folder</q-tooltip></q-btn>
                      <q-btn flat dense round size="sm" icon="delete_outline" color="negative"
                             @click.stop="$parent.$emit('row_action', {action: 'delete', row: props.row})">
                        <q-tooltip>Delete</q-tooltip></q-btn>
                    </q-td>''')
                footer = ui.label().classes('text-xs rulz-muted px-2')

            if not is_bucket:
                with ui.tab_panel('history').classes('p-0'):
                    history_box = ui.column().classes('w-full')

    # ======================================================================== helpers
    def selected_entries() -> list[hub.Entry]:
        return [state['entries'][r['path']] for r in table.selected if r['path'] in state['entries']]

    def vast_source() -> vast.Source:
        picked = selected_entries()
        if picked:
            return vast.Source.from_entries(kind, repo_id, fs.revision, picked)
        folder = state['folder']
        if folder:
            return vast.Source(kind, repo_id, fs.revision, [folder], [folder])
        return vast.Source(kind, repo_id, fs.revision)

    def update_buttons() -> None:
        n = len(table.selected)
        vast_btn.set_text(f'To vast.ai ({n})' if n else ('To vast.ai (this folder)' if state['folder']
                                                          else 'To vast.ai (whole repo)'))
        for b in (move_btn, copy_btn, copy_to_btn, delete_btn):
            b.set_enabled(n > 0)
        rename_btn.set_enabled(n == 1)
        dl_btn.set_text(f'Download ({n})' if n else 'Download all here')
        total = sum(r.get('size') or 0 for r in table.rows if not r['is_dir'])
        sel_size = sum(r.get('size') or 0 for r in table.selected if not r['is_dir'])
        nd = sum(1 for r in table.rows if r['is_dir'])
        nf = len(table.rows) - nd
        text = f'{nd} folder(s), {nf} file(s), {hub.human_size(total)} in this folder'
        if n:
            text += f'  |  {n} selected' + (f' ({hub.human_size(sel_size)})' if sel_size else '')
        footer.text = text

    table.on_select(lambda _: update_buttons())

    def render_crumbs() -> None:
        crumbs.clear()
        with crumbs:
            ui.button(repo_id.split('/')[-1], icon='home', on_click=lambda: navigate('')) \
                .props('flat dense no-caps size=sm')
            acc = ''
            for part in [p for p in state['folder'].split('/') if p]:
                acc = hub.join(acc, part)
                ui.icon('chevron_right', size='xs').classes('rulz-muted')
                ui.button(part, on_click=lambda a=acc: navigate(a)).props('flat dense no-caps size=sm')
            ui.space()
            ui.label(fs.uri(state['folder'])).classes('text-xs rulz-mono rulz-muted ellipsis')
            ui.button(icon='content_copy', on_click=lambda: (ui.clipboard.write(fs.uri(state['folder'])),
                                                               ui.notify('Copied hf:// path'))) \
                .props('flat round dense size=xs').tooltip('Copy hf:// path (works with the hf CLI and Python)')

    def to_row(e: hub.Entry, virtual: bool = False) -> dict:
        storage = ''
        if not e.is_dir and not is_bucket:
            storage = 'Xet' if e.xet_hash else ('LFS' if e.lfs else '')
        return {
            'path': e.path, 'name': e.name, 'is_dir': e.is_dir, 'size': e.size, 'icon': file_icon(e.name, e.is_dir),
            'modified': fmt_date(e.modified), 'modified_ago': fmt_ago(e.modified), 'storage': storage,
            'virtual': virtual, 'previewable': is_text_file(e.name) or _suffix(e.name) in IMAGE_EXT,
        }

    async def refresh(keep_selection: bool = False) -> None:
        folder = state['folder']
        table.props('loading')
        try:
            entries = await run.io_bound(fs.list_dir, folder)
        except Exception as e:  # noqa: BLE001
            entries = []
            missing = isinstance(e, (hub.hf_errors.EntryNotFoundError, hub.hf_errors.RemoteEntryNotFoundError))
            if not (missing and folder in state['virtual']):
                notify_error(e)
                if missing and folder:
                    table.props(remove='loading')
                    await navigate(folder.rsplit('/', 1)[0] if '/' in folder else '')
                    return
        finally:
            table.props(remove='loading')
        if not settings.show_hidden:
            entries = [e for e in entries if not e.name.startswith('.')]
        state['entries'] = {e.path: e for e in entries}
        rows = [to_row(e) for e in entries]
        for v in sorted(state['virtual']):
            if posixpath.dirname(v) == folder:
                if v not in state['entries']:
                    ve = hub.Entry(v, True)
                    state['entries'][v] = ve
                    rows.append(to_row(ve, virtual=True))
        rows.sort(key=lambda r: (not r['is_dir'], r['name'].lower()))
        has_dates = any(r['modified'] for r in rows)
        table.columns[2]['classes'] = '' if has_dates else 'hidden'
        table.columns[2]['headerClasses'] = '' if has_dates else 'hidden'
        prev = {r['path'] for r in table.selected} if keep_selection else set()
        table.update_rows(rows)
        if prev:
            table.selected[:] = [r for r in rows if r['path'] in prev]
        table.update()
        virtual_note.set_visibility(folder in state['virtual'] and not entries)
        update_buttons()

    async def navigate(folder: str) -> None:
        state['folder'] = folder.strip('/')
        table.selected.clear()
        search.value = ''
        render_crumbs()
        ui.navigate.history.replace(current_url())
        await refresh()

    def after_change(job: Job | None = None) -> None:
        # Called inside this tab's context: timers give each reload its own UI context.
        ui.timer(0.05, refresh, once=True)
        ui.timer(0.3, load_summary, once=True)

    async def open_entry(row: dict) -> None:
        if row['is_dir']:
            await navigate(row['path'])
        else:
            await preview(state['entries'][row['path']])

    table.on('open_entry', lambda e: open_entry(e.args))

    async def row_action(e: events.GenericEventArguments) -> None:
        action, row = e.args['action'], e.args['row']
        entry = state['entries'].get(row['path'])
        if not entry:
            return
        if action == 'preview':
            await preview(entry)
        elif action == 'rename':
            await rename_entry(entry)
        elif action == 'download':
            start_download([entry], via_browser=False)
        elif action == 'delete':
            await delete_entries([entry])
    table.on('row_action', row_action)

    # ======================================================================== summary / branches / history
    async def load_summary() -> None:
        try:
            s = await run.io_bound(hub.repo_summary, kind, repo_id)
        except Exception as e:  # noqa: BLE001
            notify_error(e)
            return
        state['summary'] = s
        badges.clear()
        with badges:
            private = s.get('private')
            with ui.element('span').classes('rz-vis ' + ('private' if private else 'public')):
                ui.icon('lock' if private else 'public')
                ui.label('private' if private else 'public')
            if s.get('size') is not None:
                ui.chip(hub.human_size(s['size']), icon='sd_storage').props('dense square outline')
            if s.get('files') is not None:
                ui.chip(f'{s["files"]:,} files', icon='description').props('dense square outline')
            if s.get('downloads') is not None:
                ui.chip(f'{s["downloads"]:,}', icon='download').props('dense square outline').tooltip('Downloads (30 days)')
            if s.get('likes'):
                ui.chip(f'{s["likes"]}', icon='favorite_border').props('dense square outline')
            if s.get('gated'):
                ui.chip(f'gated: {s["gated"]}', icon='how_to_reg').props('dense square outline')
            if s.get('updated'):
                ui.chip(f'updated {fmt_ago(s["updated"])}', icon='update').props('dense square outline')

    async def load_branches() -> None:
        try:
            names = await run.io_bound(fs.branches)
        except Exception:  # noqa: BLE001
            return
        current = fs.revision or 'main'
        if current not in names:
            names.append(current)
        branch_select.set_options(names, value=current)
        branch_select.on_value_change(lambda e: ui.navigate.to(browse_url(kind, repo_id, '', e.value
                                                                          if e.value != 'main' else None)))

    async def new_branch() -> None:
        name_ = await prompt('Create branch', 'Branch name', ok_label='Create',
                             hint=f'Starts from "{fs.revision or "main"}".')
        if not name_:
            return
        try:
            await run.io_bound(fs.create_branch, name_, fs.revision)
        except Exception as e:  # noqa: BLE001
            notify_error(e)
            return
        notify_ok(f'Branch "{name_}" created')
        ui.navigate.to(browse_url(kind, repo_id, '', name_))

    async def load_history() -> None:
        history_box.clear()
        with history_box:
            ui.spinner().classes('self-center my-6')
        try:
            commits = await run.io_bound(fs.history)
        except Exception as e:  # noqa: BLE001
            history_box.clear()
            notify_error(e)
            return
        history_box.clear()
        base = hub.web_url(kind, repo_id)
        rows = [{**c, 'date_str': fmt_date(c['date']), 'ago': fmt_ago(c['date']), 'url': f'{base}/commit/{c["id"]}'}
                for c in commits]
        with history_box:
            t = ui.table(columns=[
                {'name': 'title', 'label': 'Commit', 'field': 'title', 'align': 'left'},
                {'name': 'authors', 'label': 'Author', 'field': 'authors', 'align': 'left'},
                {'name': 'ago', 'label': 'When', 'field': 'ago', 'align': 'left'},
                {'name': 'short', 'label': 'Id', 'field': 'short', 'align': 'left'},
            ], rows=rows, row_key='id', pagination={'rowsPerPage': 50}).props('flat bordered').classes('w-full rulz-table')
            t.add_slot('body-cell-short', r'''
                <q-td :props="props"><a :href="props.row.url" target="_blank" class="rulz-mono text-primary">
                  {{ props.row.short }}</a></q-td>''')
            t.add_slot('body-cell-ago', r'''<q-td :props="props"><span :title="props.row.date_str">
                {{ props.row.ago }}</span></q-td>''')

    if not is_bucket:
        loaded_history = {'done': False}

        async def on_tab(e: events.ValueChangeEventArguments) -> None:
            if e.value == 'history' and not loaded_history['done']:
                loaded_history['done'] = True
                await load_history()
        panels.on_value_change(on_tab)

    # ======================================================================== folder ops
    async def new_folder() -> None:
        name_ = await prompt('New folder', 'Folder name', ok_label='Create',
                             hint='Tip: use "a/b/c" to create nested folders. Hub folders exist once they contain '
                                  'a file, so upload something into it next.',
                             validation=lambda v: _validation(v))
        if not name_:
            return
        target = hub.join(state['folder'], name_)
        parts = target.split('/')
        for i in range(len(parts)):
            state['virtual'].add('/'.join(parts[:i + 1]))
        await navigate(target)

    def _validation(v: str) -> str | None:
        try:
            hub.validate_name(v)
        except ValueError as e:
            return str(e)
        return None

    # ======================================================================== upload
    async def confirm_upload(pairs: list[tuple[str | Path, str]], source_label: str) -> tuple[bool, str, bool]:
        """Show a summary of what will be uploaded. Returns (ok, commit message, create_pr)."""
        try:
            conflicts = await run.io_bound(fs.existing, [d for _, d in pairs])
        except Exception:  # noqa: BLE001
            conflicts = set()
        total = sum(Path(s).stat().st_size for s, _ in pairs if isinstance(s, (str, Path)) and Path(s).exists())
        dest = state['folder'] or '(root)'
        with ui.dialog() as dialog, ui.card().classes('rulz-card w-[620px] max-w-[95vw] gap-3'):
            ui.label('Upload').classes('text-lg font-semibold')
            ui.label(f'{len(pairs)} file(s), {hub.human_size(total)} from {source_label}')
            ui.label(f'Destination: {repo_id} / {dest}').classes('rulz-mono text-sm')
            with ui.scroll_area().classes('max-h-48 w-full rz-box'):
                for _, d in pairs[:300]:
                    with ui.row().classes('items-center gap-2 no-wrap'):
                        ui.icon('sync_problem' if d in conflicts else 'add', size='xs',
                                color='warning' if d in conflicts else 'positive')
                        ui.label(d).classes('rulz-mono text-sm')
                if len(pairs) > 300:
                    ui.label(f'... and {len(pairs) - 300} more').classes('text-sm rulz-muted')
            if conflicts:
                with ui.row().classes('items-center gap-2 rz-warn no-wrap'):
                    ui.icon('warning')
                    ui.label(f'{len(conflicts)} file(s) already exist and will be overwritten.')
            msg, pr = commit_fields(fs, f'Upload {len(pairs)} file(s)' if len(pairs) > 1 else
                                    f'Upload {pairs[0][1]}')
            with ui.row().classes('w-full justify-end gap-2'):
                ui.button('Cancel', on_click=lambda: dialog.submit(False)).props('flat')
                ui.button('Upload', icon='upload', on_click=lambda: dialog.submit(True))
        ok = await dialog
        return bool(ok), (msg.value if msg else '') or 'Upload files', bool(pr.value) if pr else False

    async def upload_from_computer() -> None:
        paths = await pick_local('any', 'Upload: choose files and folders')
        if not paths:
            return
        dest_folder = state['folder']
        pairs = await run.io_bound(hub.local_items_to_uploads, paths, dest_folder)
        if not pairs:
            ui.notify('Nothing to upload (empty folders?)', type='warning')
            return
        ok, message, create_pr = await confirm_upload(pairs, 'this computer')
        if not ok:
            return
        run_job(f'Upload {len(pairs)} file(s) to {repo_id}', lambda job: fs.upload(pairs, message, create_pr),
                icon='upload', success=f'Uploaded {len(pairs)} file(s) to {repo_id}', after=after_change)

    async def upload_from_browser() -> None:
        tmp = Path(tempfile.mkdtemp(prefix='rulz-hf-up-'))
        staged: dict[str, Path] = {}
        dest_folder = state['folder']

        with ui.dialog() as dialog, ui.card().classes('rulz-card w-[640px] max-w-[95vw] gap-3'):
            ui.label('Upload through the browser').classes('text-lg font-semibold')
            ui.label(f'Destination: {repo_id} / {dest_folder or "(root)"}').classes('rulz-mono text-sm')

            async def on_upload(e: events.UploadEventArguments) -> None:
                filename = Path(e.file.name).name or 'file'
                target = tmp / filename
                await e.file.save(target)
                staged[filename] = target
                count.set_text(f'{len(staged)} file(s) ready')
                go.set_enabled(True)

            ui.upload(multiple=True, auto_upload=True, on_upload=on_upload,
                      label='Drop files here or click +').props('flat bordered color=primary').classes('w-full rulz-drop')
            ui.label('For very large files (several GB) or whole folders use "Files or folders from this computer" '
                     'instead: it reads directly from disk without copying through the browser.') \
                .classes('text-xs rulz-muted')
            with ui.row().classes('w-full items-center gap-2'):
                count = ui.label('No files yet').classes('text-sm rulz-muted')
                ui.space()
                ui.button('Cancel', on_click=lambda: dialog.submit(False)).props('flat')
                go = ui.button('Continue', icon='arrow_forward', on_click=lambda: dialog.submit(True))
                go.disable()
        ok = await dialog
        if not ok or not staged:
            shutil.rmtree(tmp, ignore_errors=True)
            return
        pairs = [(p, hub.join(dest_folder, n)) for n, p in staged.items()]
        ok, message, create_pr = await confirm_upload(pairs, 'the browser')
        if not ok:
            shutil.rmtree(tmp, ignore_errors=True)
            return

        def work(job: Job) -> None:
            try:
                fs.upload(pairs, message, create_pr)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        run_job(f'Upload {len(pairs)} file(s) to {repo_id}', work, icon='upload',
                success=f'Uploaded {len(pairs)} file(s)', after=after_change)

    # ======================================================================== sync
    async def sync_dialog(direction: str = 'up') -> None:
        if not is_bucket:
            await push_folder_dialog()
            return
        with ui.dialog() as dialog, ui.card().classes('rulz-card w-[760px] max-w-[96vw] gap-3'):
            ui.label('Sync folder with bucket').classes('text-lg font-semibold')
            ui.label('Like rsync: only new or changed files are transferred.').classes('text-sm rulz-muted')
            dir_toggle = ui.toggle({'up': 'This computer  >  bucket', 'down': 'Bucket  >  this computer'},
                                   value=direction).props('no-caps spread unelevated toggle-color=primary')
            with ui.row().classes('w-full items-center no-wrap gap-2'):
                local = ui.input('Local folder').props('outlined dense').classes('flex-1 rulz-mono')

                async def browse_local() -> None:
                    r = await pick_local('folder', 'Choose the local folder', local.value or None)
                    if r:
                        local.value = r[0]
                ui.button(icon='folder_open', on_click=browse_local).props('flat round')
            remote = ui.input('Bucket folder', value=state['folder']).props('outlined dense prefix="/"') \
                .classes('w-full rulz-mono')
            with ui.row().classes('w-full gap-3'):
                include = ui.input('Include patterns', placeholder='*.safetensors, *.json') \
                    .props('outlined dense').classes('flex-1')
                exclude = ui.input('Exclude patterns', placeholder='*.tmp, logs/*').props('outlined dense').classes(
                    'flex-1')
            delete = ui.checkbox('Delete files at the destination that are not in the source (mirror)')
            result_box = ui.column().classes('w-full gap-1')
            with ui.row().classes('w-full justify-end gap-2'):
                ui.button('Close', on_click=lambda: dialog.submit(None)).props('flat')
                preview_btn = ui.button('Preview changes', icon='preview').props('outline')
                run_btn = ui.button('Run sync', icon='sync')
                run_btn.disable()

        plan_state: dict = {'file': None}

        def patterns(v: str | None) -> list[str]:
            return [p.strip() for p in (v or '').split(',') if p.strip()]

        def invalidate() -> None:
            plan_state['file'] = None
            run_btn.disable()
            result_box.clear()
        for el in (dir_toggle, local, remote, include, exclude, delete):
            el.on_value_change(invalidate)

        async def do_preview() -> None:
            from ..platform_utils import to_local_path
            local_dir = to_local_path(local.value or '')
            if not local_dir:
                notify_error('Choose a local folder first.')
                return
            if dir_toggle.value == 'up' and not Path(local_dir).is_dir():
                notify_error(f'Local folder not found: {local_dir}')
                return
            plan_file = str(Path(tempfile.mkdtemp(prefix='rulz-hf-plan-')) / 'plan.jsonl')
            preview_btn.props('loading')
            try:
                plan = await run.io_bound(fs.sync, local_dir, remote.value or '', dir_toggle.value, delete.value,
                                          patterns(include.value), patterns(exclude.value), plan_file)
            except Exception as e:  # noqa: BLE001
                notify_error(e)
                return
            finally:
                preview_btn.props(remove='loading')
            s = plan.summary()
            result_box.clear()
            with result_box:
                with ui.row().classes('gap-2'):
                    for key, icon, color in (('uploads', 'upload', 'primary'), ('downloads', 'download', 'primary'),
                                             ('deletes', 'delete', 'negative'), ('skips', 'check', 'grey')):
                        if s.get(key):
                            ui.chip(f'{s[key]} {key}', icon=icon, color=color, text_color='white').props('dense')
                    ui.chip(f'{hub.human_size(s.get("total_size", 0))} to transfer', icon='swap_vert') \
                        .props('dense outline')
                ops = [op for op in plan.operations if op.action != 'skip']
                if ops:
                    ui.table(columns=[
                        {'name': 'action', 'label': 'Action', 'field': 'action', 'align': 'left'},
                        {'name': 'path', 'label': 'Path', 'field': 'path', 'align': 'left'},
                        {'name': 'size', 'label': 'Size', 'field': 'size', 'align': 'right', ':format': SIZE_FORMAT_JS},
                        {'name': 'reason', 'label': 'Why', 'field': 'reason', 'align': 'left'},
                    ], rows=[{'id': i, 'action': op.action, 'path': op.path, 'size': op.size, 'reason': op.reason}
                             for i, op in enumerate(ops)], row_key='id', pagination={'rowsPerPage': 8}) \
                        .props('flat dense bordered').classes('w-full rulz-table')
                else:
                    ui.label('Everything is already in sync.').classes('rz-ok')
            plan_state['file'] = plan_file if ops else None
            run_btn.set_enabled(bool(ops))

        def do_run() -> None:
            plan_file = plan_state['file']
            if not plan_file:
                return
            dialog.submit(None)
            down = dir_toggle.value == 'down'
            from ..platform_utils import to_local_path
            local_dir = Path(to_local_path(local.value or ''))

            def work(job: Job) -> Path | None:
                fs.sync('', '', 'up', False, None, None, apply_file=plan_file)
                return local_dir if down else None
            run_job(f'Sync {repo_id}', work, icon='sync', success='Sync finished', after=after_change)
        preview_btn.on_click(do_preview)
        run_btn.on_click(do_run)
        await dialog

    async def push_folder_dialog() -> None:
        with ui.dialog() as dialog, ui.card().classes('rulz-card w-[680px] max-w-[95vw] gap-3'):
            ui.label('Push a whole folder').classes('text-lg font-semibold')
            ui.label('Uploads the contents of a local folder into the current repo folder in a single commit. '
                     'Unchanged large files are de-duplicated by the Hub, so re-pushing is fast.') \
                .classes('text-sm rulz-muted')
            with ui.row().classes('w-full items-center no-wrap gap-2'):
                local = ui.input('Local folder').props('outlined dense').classes('flex-1 rulz-mono')

                async def browse_local() -> None:
                    r = await pick_local('folder', 'Choose the folder to push', local.value or None)
                    if r:
                        local.value = r[0]
                ui.button(icon='folder_open', on_click=browse_local).props('flat round')
            remote = ui.input('Into repo folder', value=state['folder']).props('outlined dense prefix="/"') \
                .classes('w-full rulz-mono')
            mirror = ui.checkbox('Mirror: delete files in that repo folder that are not in the local folder')
            msg, pr = commit_fields(fs, 'Upload folder')
            with ui.row().classes('w-full justify-end gap-2'):
                ui.button('Cancel', on_click=lambda: dialog.submit(False)).props('flat')
                ui.button('Push', icon='upload', on_click=lambda: dialog.submit(True))
        if not await dialog:
            return
        from ..platform_utils import to_local_path
        local_dir = Path(to_local_path(local.value or ''))
        if not local_dir.is_dir():
            notify_error(f'Local folder not found: {local_dir}')
            return
        if mirror.value and not await confirm('Mirror the folder?', 'Files in the repo folder that do not exist '
                                              'locally will be deleted in the same commit.', ok_label='Mirror',
                                              ok_color='warning'):
            return
        dest, message, create_pr, mirror_v = (remote.value or '').strip('/'), msg.value or 'Upload folder', \
            bool(pr.value), bool(mirror.value)
        run_job(f'Push {local_dir.name} to {repo_id}',
                lambda job: fs.upload_folder(local_dir, dest, message, mirror_v, create_pr),
                icon='upload', success='Folder pushed', after=after_change)

    # ======================================================================== download
    def start_download(entries: list[hub.Entry], via_browser: bool) -> None:
        if not entries:
            return
        label = entries[0].name if len(entries) == 1 else f'{len(entries)} items'
        if via_browser:
            def work(job: Job):
                return fs.download_to_temp(entries, job)

            def deliver(job: Job) -> None:
                path, filename = job.result
                _temp_downloads.append(path)
                ui.download.file(path, filename)
                job.result = None
            run_job(f'Prepare {label} for the browser', work, icon='download',
                    success=f'{label} is downloading in your browser', after=deliver)
            return
        target = settings.effective_download_dir() / f'{kind}s' / repo_id.split('/')[0] / repo_id.split('/')[1]

        def work_local(job: Job) -> Path:
            fs.download(entries, target, job)
            return target
        run_job(f'Download {label}', work_local, icon='download', success=lambda job: f'Saved to {job.result}')

    async def download_selected(via_browser: bool) -> None:
        entries = selected_entries() or [e for e in state['entries'].values() if e.raw is not None]
        if not entries:
            ui.notify('Nothing to download here')
            return
        start_download(entries, via_browser)

    # ======================================================================== rename / move / copy
    async def rename_entry(entry: hub.Entry) -> None:
        new_name = await prompt(f'Rename {"folder" if entry.is_dir else "file"}', 'New name', entry.name,
                                ok_label='Rename', select_stem=not entry.is_dir,
                                hint='You can also type a path like "subfolder/new-name" to move it.',
                                validation=_validation)
        if not new_name or new_name == entry.name:
            return
        new_path = hub.join(entry.parent, new_name)
        await execute_moves([(entry, new_path)], copy=False, title=f'Rename {entry.name} to {new_name}')

    async def rename_selected() -> None:
        sel = selected_entries()
        if len(sel) == 1:
            await rename_entry(sel[0])

    async def move_selected(copy: bool) -> None:
        sel = selected_entries()
        if not sel:
            return
        dest = await pick_remote_folder(fs, state['folder'], f'{"Copy" if copy else "Move"} {len(sel)} item(s) to...',
                                        disabled={e.path for e in sel if e.is_dir} if not copy else set())
        if dest is None:
            return
        pairs = [(e, hub.join(dest, e.name)) for e in sel]
        await execute_moves(pairs, copy=copy,
                            title=f'{"Copy" if copy else "Move"} {len(sel)} item(s) to /{dest}')

    async def execute_moves(pairs: list[tuple[hub.Entry, str]], copy: bool, title: str) -> None:
        if any(e.raw is None for e, _ in pairs):
            notify_error('That folder is new and empty on the Hub; there is nothing to move yet.')
            return
        n = ui.notification('Planning...', spinner=True, timeout=None)
        try:
            plan = await run.io_bound(fs.plan_moves, pairs)
            conflicts = await run.io_bound(fs.existing, [d for _, d in plan])
        except Exception as e:  # noqa: BLE001
            notify_error(e)
            return
        finally:
            n.dismiss()
        if not plan:
            ui.notify('Nothing to do')
            return
        verb = 'Copy' if copy else 'Move'
        with ui.dialog() as dialog, ui.card().classes('rulz-card w-[640px] max-w-[95vw] gap-3'):
            ui.label(title).classes('text-lg font-semibold')
            ui.label(f'{len(plan)} file(s) will be {"copied" if copy else "moved"}'
                     + (' server-side (no download / re-upload).' if is_bucket else ' in one commit.'))
            with ui.scroll_area().classes('max-h-56 w-full rz-box'):
                for f, d in plan[:300]:
                    with ui.row().classes('items-center gap-1 no-wrap text-sm rulz-mono'):
                        ui.label(f.path).classes('rulz-muted')
                        ui.icon('arrow_forward', size='xs')
                        ui.label(d).classes('rz-warn' if d in conflicts else '')
                if len(plan) > 300:
                    ui.label(f'... and {len(plan) - 300} more').classes('text-sm rulz-muted')
            if conflicts:
                with ui.row().classes('items-center gap-2 rz-warn no-wrap'):
                    ui.icon('warning')
                    ui.label(f'{len(conflicts)} destination file(s) already exist and will be overwritten.')
            msg, pr = commit_fields(fs, title)
            with ui.row().classes('w-full justify-end gap-2'):
                ui.button('Cancel', on_click=lambda: dialog.submit(False)).props('flat')
                ui.button(verb, icon='file_copy' if copy else 'drive_file_move', on_click=lambda: dialog.submit(True))
        if not await dialog:
            return
        message, create_pr = (msg.value if msg else '') or title, bool(pr.value) if pr else False
        op = fs.copy if copy else fs.move
        run_job(title, lambda job: op(plan, message, create_pr), icon='drive_file_move',
                success=f'{verb} complete ({len(plan)} file(s))', after=after_change)

    async def copy_to_dialog() -> None:
        sel = selected_entries()
        if not sel:
            return
        ident = None
        try:
            ident = await run.io_bound(hub.whoami)
        except Exception:  # noqa: BLE001
            pass
        options: list[str] = []
        if ident:
            try:
                rows = await run.io_bound(hub.list_namespace, ident.name)
                options = sorted(f'{r.kind}:{r.id}' for r in rows)
            except Exception:  # noqa: BLE001
                options = []
        with ui.dialog() as dialog, ui.card().classes('rulz-card w-[620px] max-w-[95vw] gap-3'):
            ui.label(f'Copy {len(sel)} item(s) to another repo or bucket').classes('text-lg font-semibold')
            ui.label('Copies run on the Hub servers (no download/upload) whenever possible. '
                     'Bucket -> repo copies go through this computer.').classes('text-sm rulz-muted')
            dest_kind = ui.toggle({k: hub.KIND_LABEL[k] for k in hub.KINDS}, value='bucket') \
                .props('no-caps spread unelevated toggle-color=primary')
            if options:
                quick = ui.select({o: f'{hub.KIND_LABEL[o.split(":")[0]]}:  {o.split(":", 1)[1]}' for o in options},
                                  label='Pick one of yours (optional)', with_input=True, clearable=True) \
                    .props('outlined dense options-dense').classes('w-full')
            dest_id = ui.input('Destination id (namespace/name)').props('outlined dense').classes('w-full rulz-mono')

            if options:
                def on_pick(e) -> None:
                    if e.value and ':' in str(e.value):
                        k, i = str(e.value).split(':', 1)
                        dest_kind.value = k
                        dest_id.value = i
                quick.on_value_change(on_pick)
            dest_folder = ui.input('Destination folder', value='').props('outlined dense prefix="/"') \
                .classes('w-full rulz-mono')
            dest_rev = ui.input('Branch', value='main').props('outlined dense').classes('w-40')
            dest_rev.bind_visibility_from(dest_kind, 'value', backward=lambda v: v != 'bucket')
            with ui.row().classes('w-full justify-end gap-2'):
                ui.button('Cancel', on_click=lambda: dialog.submit(False)).props('flat')
                ui.button('Copy', icon='move_up', on_click=lambda: dialog.submit(True))
        if not await dialog:
            return
        target = str(dest_id.value or '').strip().strip('/')
        if not hub._ID_RE.match(target):
            notify_error('Enter a destination like "namespace/name".')
            return
        dk, folder_, drev = dest_kind.value, dest_folder.value or '', (dest_rev.value or None)
        if dk == kind and target == repo_id:
            notify_error('Use "Copy" (not "Copy to...") to copy inside the same repo.')
            return
        run_job(f'Copy {len(sel)} item(s) to {target}',
                lambda job: fs.copy_to(sel, dk, target, folder_, drev if dk != 'bucket' else None, job=job),
                icon='move_up', success=f'Copied to {hub.KIND_LABEL[dk].lower()} {target}')

    # ======================================================================== delete
    async def delete_entries(entries: list[hub.Entry]) -> None:
        if not entries:
            return
        virtual = [e for e in entries if e.raw is None and e.is_dir]
        for v in virtual:
            state['virtual'] = {p for p in state['virtual'] if p != v.path and not p.startswith(v.path + '/')}
        entries = [e for e in entries if e not in virtual]
        if not entries:
            await refresh()
            return
        items = [f'{e.path}/' if e.is_dir else e.path for e in entries]
        with ui.dialog() as dialog, ui.card().classes('rulz-card w-[560px] max-w-[95vw] gap-3'):
            ui.label(f'Delete {len(entries)} item(s)?').classes('text-lg font-semibold')
            with ui.scroll_area().classes('max-h-48 w-full rz-box'):
                for it in items:
                    ui.label(it).classes('rulz-mono text-sm')
            if any(e.is_dir for e in entries):
                ui.label('Folders are deleted with everything inside them.').classes('text-sm rz-warn')
            if is_bucket:
                ui.label('Buckets have no history: deleted files cannot be recovered.').classes('text-sm rz-err')
            else:
                ui.label('Old versions stay in the git history of the repo.').classes('text-sm rulz-muted')
            msg, pr = commit_fields(fs, f'Delete {items[0]}' if len(items) == 1 else f'Delete {len(items)} items')
            with ui.row().classes('w-full justify-end gap-2'):
                ui.button('Cancel', on_click=lambda: dialog.submit(False)).props('flat')
                ui.button('Delete', icon='delete', color='negative', on_click=lambda: dialog.submit(True))
        if not await dialog:
            return
        message, create_pr = (msg.value if msg else '') or 'Delete files', bool(pr.value) if pr else False
        run_job(f'Delete {len(entries)} item(s) from {repo_id}', lambda job: fs.delete(entries, message, create_pr),
                icon='delete', success='Deleted', after=after_change)

    async def delete_selected() -> None:
        await delete_entries(selected_entries())

    # ======================================================================== preview / edit
    async def preview(entry: hub.Entry) -> None:
        suffix = _suffix(entry.name)
        image = suffix in IMAGE_EXT
        text = is_text_file(entry.name)
        if not image and not text:
            with ui.dialog() as d, ui.card().classes('rulz-card gap-3'):
                ui.label(entry.name).classes('text-lg font-semibold')
                ui.label(f'{hub.human_size(entry.size)}. No preview for this file type.')
                with ui.row().classes('justify-end w-full gap-2'):
                    ui.button('Close', on_click=d.close).props('flat')
                    ui.button('Download', icon='download', on_click=lambda: (d.close(),
                                                                             start_download([entry], False)))
            d.open()
            return
        if (entry.size or 0) > PREVIEW_LIMIT:
            notify_error(f'{entry.name} is {hub.human_size(entry.size)}; too large to preview. Download it instead.')
            return
        n = ui.notification(f'Loading {entry.name}...', spinner=True, timeout=None)
        try:
            data = await run.io_bound(fs.read_bytes, entry.path)
        except Exception as e:  # noqa: BLE001
            notify_error(e)
            return
        finally:
            n.dismiss()

        with ui.dialog() as dialog, \
                ui.card().classes('rulz-card w-[1100px] max-w-[96vw] h-[85vh] p-0 gap-0 column no-wrap'):
            with ui.row().classes('w-full items-center px-4 py-2 gap-2 no-wrap'):
                ui.icon(file_icon(entry.name, False), color='primary')
                ui.label(entry.path).classes('text-base font-medium rulz-mono ellipsis')
                ui.label(hub.human_size(entry.size)).classes('text-xs rulz-muted')
                ui.space()
                edit_btn = ui.button('Edit', icon='edit').props('flat no-caps')
                save_btn = ui.button('Save', icon='save').props('unelevated no-caps')
                save_btn.set_visibility(False)
                ui.button(icon='download', on_click=lambda: start_download([entry], True)) \
                    .props('flat round').tooltip('Download through the browser')
                ui.button(icon='close', on_click=dialog.close).props('flat round')
            ui.separator()
            body = ui.column().classes('w-full flex-1 overflow-auto p-0 gap-0')

        if image:
            edit_btn.set_visibility(False)
            mime = 'image/svg+xml' if suffix == '.svg' else f'image/{suffix[1:].replace("jpg", "jpeg")}'
            with body:
                ui.image(f'data:{mime};base64,{base64.b64encode(data).decode()}') \
                    .props('fit=contain').classes('w-full h-full')
            dialog.open()
            return

        try:
            content = data.decode('utf-8')
        except UnicodeDecodeError:
            content = data.decode('latin-1')
        dark = config.get().dark_mode != 'light'
        lang = LANGUAGE.get(suffix) or ('Markdown' if entry.name.lower().startswith('readme') else None)
        editor_state: dict = {'editor': None}

        def show_view() -> None:
            body.clear()
            with body:
                if lang == 'Markdown':
                    with ui.column().classes('w-full p-6 max-w-[900px] mx-auto'):
                        ui.markdown(_strip_front_matter(content), extras=['fenced-code-blocks', 'tables'])
                else:
                    pretty = content
                    if suffix == '.json':
                        try:
                            pretty = json.dumps(json.loads(content), indent=2, ensure_ascii=False)
                        except ValueError:
                            pass
                    viewer = ui.codemirror(pretty, language=lang, theme='githubDark' if dark else 'githubLight',
                                           line_wrapping=True).classes('w-full h-full').style('min-height: 100%')
                    viewer.disable()

        def start_edit() -> None:
            body.clear()
            with body:
                editor_state['editor'] = ui.codemirror(content, language=lang,
                                                       theme='githubDark' if dark else 'githubLight',
                                                       line_wrapping=True).classes('w-full h-full') \
                    .style('min-height: 100%')
            edit_btn.set_visibility(False)
            save_btn.set_visibility(True)

        async def save() -> None:
            editor = editor_state['editor']
            if editor is None:
                return
            new_content = editor.value
            if new_content == content:
                ui.notify('No changes')
                return
            message = f'Update {entry.path}'
            create_pr = False
            if not is_bucket:
                with ui.dialog() as cd, ui.card().classes('rulz-card w-[520px] gap-3'):
                    ui.label('Save changes').classes('text-lg font-semibold')
                    m, p = commit_fields(fs, message)
                    with ui.row().classes('w-full justify-end gap-2'):
                        ui.button('Cancel', on_click=lambda: cd.submit(False)).props('flat')
                        ui.button('Commit', icon='save', on_click=lambda: cd.submit(True))
                if not await cd:
                    return
                message, create_pr = (m.value if m else '') or message, bool(p.value) if p else False
            save_btn.props('loading')
            try:
                await run.io_bound(fs.write_bytes, entry.path, new_content.encode('utf-8'), message, create_pr)
            except Exception as e:  # noqa: BLE001
                notify_error(e)
                return
            finally:
                save_btn.props(remove='loading')
            notify_ok('Opened a pull request with your changes' if create_pr else 'Saved')
            dialog.close()
            after_change()

        edit_btn.on_click(start_edit)
        save_btn.on_click(save)
        show_view()
        dialog.open()

    # ======================================================================== initial load
    render_crumbs()

    # Separate timers: each runs concurrently with its own UI context.
    ui.timer(0.01, refresh, once=True)
    ui.timer(0.01, load_summary, once=True)
    if not is_bucket:
        ui.timer(0.01, load_branches, once=True)


def _strip_front_matter(text: str) -> str:
    """Model cards start with a YAML block; show it as a code block instead of a broken table."""
    if text.startswith('---'):
        end = text.find('\n---', 3)
        if end != -1:
            yaml_part = text[3:end].strip('\n')
            return f'```yaml\n{yaml_part}\n```\n' + text[end + 4:]
    return text
