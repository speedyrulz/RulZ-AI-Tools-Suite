"""Dialogs for picking local files/folders (on this computer) and remote folders (in a repo/bucket)."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from nicegui import run, ui

from .. import hub
from ..platform_utils import quick_locations, to_local_path
from .common import SIZE_FORMAT_JS, fmt_date, notify_error

_last_local_dir: Path | None = None


_WIN_HIDDEN = 0x2 | 0x4  # FILE_ATTRIBUTE_HIDDEN | FILE_ATTRIBUTE_SYSTEM


def created_time(st: os.stat_result) -> float | None:
    """When a file was created, where the system keeps it: st_birthtime (Windows since Python 3.12, macOS), or
    st_ctime on older Windows Pythons (there it is the creation time). Linux / WSL: None (st_ctime is the last
    metadata change there, not the creation)."""
    birth = getattr(st, 'st_birthtime', None)
    if birth:
        return birth
    return st.st_ctime if os.name == 'nt' else None


def _scan(path: Path, folders_only: bool, show_hidden: bool) -> list[dict]:
    if not path.is_dir():  # here, in the worker thread: a slow or gone network drive must not stall the page
        raise NotADirectoryError(f'Not a folder: {path}')
    rows: list[dict] = []
    with os.scandir(path) as it:
        for entry in it:
            if not show_hidden and entry.name.startswith('.'):
                continue
            try:
                is_dir = entry.is_dir()
                if folders_only and not is_dir:
                    continue
                st = entry.stat()
                if not show_hidden and getattr(st, 'st_file_attributes', 0) & _WIN_HIDDEN:
                    continue
                rows.append({
                    'path': entry.path, 'name': entry.name, 'is_dir': is_dir,
                    'size': None if is_dir else st.st_size,
                    'modified': fmt_date(datetime.fromtimestamp(st.st_mtime)),
                    'created': fmt_date(datetime.fromtimestamp(c)) if (c := created_time(st)) else '',
                    'icon': 'folder' if is_dir else 'draft',
                })
            except OSError:
                continue
    rows.sort(key=lambda r: (not r['is_dir'], r['name'].lower()))
    return rows


async def pick_local(mode: str = 'any', title: str | None = None, start: str | Path | None = None) -> list[str] | None:
    """Browse this computer's file system.

    mode='any'    -> select any number of files and/or folders, returns their paths
    mode='folder' -> navigate to a folder and confirm it, returns [folder]
    """
    folders_only = mode == 'folder'
    title = title or ('Choose a folder' if folders_only else 'Choose files and folders')
    places = await run.io_bound(quick_locations)
    initial = Path(start) if start and await run.io_bound(Path(start).exists) else (_last_local_dir or Path.home())
    state = {'cwd': initial, 'hidden': False}

    with ui.dialog().props('maximized-mobile') as dialog, \
            ui.card().classes('rulz-card w-[980px] max-w-[96vw] h-[82vh] p-0 gap-0 column no-wrap'):
        with ui.row().classes('w-full items-center px-4 py-3 gap-2 no-wrap'):
            ui.icon('computer', size='sm', color='primary')
            ui.label(title).classes('text-lg font-semibold')
            ui.label('(this computer)').classes('text-sm rulz-muted')
            ui.space()
            ui.button(icon='close', on_click=lambda: dialog.submit(None)).props('flat round dense')
        ui.separator()
        with ui.row().classes('w-full flex-1 no-wrap gap-0 overflow-hidden'):
            with ui.column().classes('w-48 p-2 gap-0 overflow-auto gt-xs'):
                ui.label('Places').classes('text-xs uppercase rulz-muted px-2 pt-1')
                for label, icon, p in places:
                    ui.button(label, icon=icon, on_click=lambda p=p: go(p)) \
                        .props('flat no-caps align=left dense').classes('w-full')
            ui.separator().props('vertical')
            with ui.column().classes('flex-1 p-3 gap-2 overflow-hidden column no-wrap'):
                with ui.row().classes('w-full items-center no-wrap gap-1'):
                    ui.button(icon='arrow_upward', on_click=lambda: go(state['cwd'].parent)) \
                        .props('flat round dense').tooltip('Parent folder')
                    path_input = ui.input().props('dense outlined').classes('flex-1 rulz-mono')
                    path_input.on('keydown.enter', lambda: go(Path(to_local_path(path_input.value or ''))))
                    ui.button(icon='refresh', on_click=lambda: go(state['cwd'])).props('flat round dense')
                with ui.row().classes('w-full items-center gap-3'):
                    search = ui.input(placeholder='Filter...').props('dense outlined clearable').classes('w-60')
                    async def set_hidden(e) -> None:
                        state['hidden'] = e.value
                        await go(state['cwd'])
                    ui.checkbox('Show hidden', value=False, on_change=set_hidden).classes('text-sm')
                    if not folders_only:
                        ui.label('Tip: tick files and/or folders, then press Select.').classes('text-xs rulz-muted')
                columns = [
                    {'name': 'name', 'label': 'Name', 'field': 'name', 'align': 'left', 'sortable': True},
                    {'name': 'size', 'label': 'Size', 'field': 'size', 'align': 'right', 'sortable': True,
                     ':format': SIZE_FORMAT_JS},
                    {'name': 'created', 'label': 'Created', 'field': 'created', 'align': 'left', 'sortable': True},
                    {'name': 'modified', 'label': 'Modified', 'field': 'modified', 'align': 'left', 'sortable': True},
                ]
                table = ui.table(columns=columns, rows=[], row_key='path',
                                 selection=None if folders_only else 'multiple',
                                 pagination={'rowsPerPage': 0}) \
                    .props('flat dense virtual-scroll binary-state-sort') \
                    .classes('w-full flex-1 rulz-table').style('min-height: 0')
                table.bind_filter_from(search, 'value')
                table.add_slot('body-cell-name', r'''
                    <q-td :props="props">
                      <div class="row items-center no-wrap q-gutter-x-sm cursor-pointer rulz-name"
                           @click="$parent.$emit('enter', props.row)">
                        <q-icon :name="props.row.icon" size="20px" :class="props.row.is_dir ? 'rz-ic-dir' : 'rz-ic-file'"/>
                        <span>{{ props.row.name }}</span>
                      </div>
                    </q-td>''')
                table.on('enter', lambda e: go(Path(e.args['path'])) if e.args['is_dir'] else toggle(e.args))
        ui.separator()
        with ui.row().classes('w-full items-center px-4 py-3 gap-2 no-wrap'):
            status = ui.label().classes('text-sm rulz-muted ellipsis flex-1 min-w-0 rulz-mono')
            ui.button('Cancel', on_click=lambda: dialog.submit(None)).props('flat')
            if folders_only:
                select_btn = ui.button('Use this folder', icon='check', on_click=lambda: finish())
            else:
                select_btn = ui.button('Select', icon='check', on_click=lambda: finish())

    def toggle(row: dict) -> None:
        if any(r['path'] == row['path'] for r in table.selected):
            table.selected[:] = [r for r in table.selected if r['path'] != row['path']]
        else:
            table.selected.append(row)
        table.update()
        update_status()

    def update_status() -> None:
        if folders_only:
            status.text = str(state['cwd'])
            return
        n = len(table.selected)
        size = sum(r['size'] or 0 for r in table.selected if not r['is_dir'])
        dirs = sum(1 for r in table.selected if r['is_dir'])
        status.text = (f'{n} selected' + (f' ({dirs} folder(s))' if dirs else '') +
                       (f', {hub.human_size(size)} in files' if size else '')) if n else 'Nothing selected'
        select_btn.set_enabled(bool(n))

    table.on_select(lambda _: update_status())

    async def go(path: Path) -> None:
        global _last_local_dir
        try:
            path = path.expanduser()
            rows = await run.io_bound(_scan, path, folders_only, state['hidden'])
        except Exception as e:  # noqa: BLE001
            notify_error(e if isinstance(e, str) else str(e))
            return
        state['cwd'] = path
        _last_local_dir = path
        path_input.value = str(path)
        cls = '' if any(r['created'] for r in rows) else 'hidden'  # Linux keeps no creation date
        table.columns[2]['classes'] = table.columns[2]['headerClasses'] = cls
        table.update_rows(rows)
        update_status()

    def finish() -> None:
        if folders_only:
            dialog.submit([str(state['cwd'])])
        else:
            dialog.submit([r['path'] for r in table.selected]) if table.selected else None

    await go(state['cwd'])
    return await dialog


async def pick_remote_folder(fs: hub.RemoteFS, start: str = '', title: str = 'Choose destination folder',
                             disabled: set[str] | None = None) -> str | None:
    """Navigate the folders of a repo/bucket and return the chosen folder path ('' = root)."""
    disabled = disabled or set()
    state = {'folder': start.strip('/'), 'virtual': set()}

    with ui.dialog() as dialog, ui.card().classes('rulz-card w-[640px] max-w-[95vw] h-[70vh] p-0 gap-0 column no-wrap'):
        with ui.row().classes('w-full items-center px-4 py-3 gap-2'):
            ui.icon('drive_file_move', size='sm', color='primary')
            ui.label(title).classes('text-lg font-semibold')
            ui.space()
            ui.button(icon='close', on_click=lambda: dialog.submit(None)).props('flat round dense')
        ui.separator()
        crumbs = ui.row().classes('w-full items-center px-3 py-2 gap-0 rulz-crumb')
        listing = ui.column().classes('w-full flex-1 overflow-auto px-2 gap-0')
        ui.separator()
        with ui.row().classes('w-full items-center px-4 py-3 gap-2'):
            ui.button('New folder', icon='create_new_folder', on_click=lambda: new_folder()).props('flat no-caps')
            ui.space()
            ui.button('Cancel', on_click=lambda: dialog.submit(None)).props('flat')
            choose = ui.button('Move here', icon='check', on_click=lambda: dialog.submit(state['folder']))

    async def new_folder() -> None:
        from .common import prompt
        name = await prompt('New folder', 'Folder name', validation=_name_error)
        if name:
            path = hub.join(state['folder'], name)
            state['virtual'].add(path)
            await show(path)

    async def show(folder: str) -> None:
        state['folder'] = folder
        crumbs.clear()
        with crumbs:
            ui.button(icon='home', on_click=lambda: show('')).props('flat dense round size=sm')
            acc = ''
            for part in [p for p in folder.split('/') if p]:
                acc = hub.join(acc, part)
                ui.label('/').classes('rulz-muted')
                ui.button(part, on_click=lambda a=acc: show(a)).props('flat dense no-caps size=sm')
        listing.clear()
        with listing:
            ui.spinner().classes('self-center my-6')
        try:
            folders = [] if folder in state['virtual'] else await run.io_bound(fs.list_folders, folder)
        except Exception as e:  # noqa: BLE001
            folders = []
            notify_error(e)
        folders += [v for v in state['virtual'] if hub.join(folder, v.rsplit('/', 1)[-1]) == v and v not in folders
                    and v != folder]
        listing.clear()
        with listing:
            if not folders:
                ui.label('No subfolders here.').classes('rulz-muted text-sm p-4')
            for f in sorted(folders):
                blocked = f in disabled or any(f.startswith(d + '/') for d in disabled)
                with ui.item(on_click=(lambda f=f: show(f)) if not blocked else None).classes('rounded'):
                    with ui.item_section().props('avatar'):
                        ui.icon('folder').classes('rz-ic-muted' if blocked else 'rz-ic-dir')
                    with ui.item_section():
                        ui.item_label(f.rsplit('/', 1)[-1]).classes('rulz-muted' if blocked else '')
        blocked_here = folder in disabled or any(folder.startswith(d + '/') for d in disabled)
        choose.set_enabled(not blocked_here)
        choose.set_text(f'Choose "{folder or "/ (root)"}"')

    await show(state['folder'])
    return await dialog


def _name_error(v: str) -> str | None:
    try:
        hub.validate_name(v)
    except ValueError as e:
        return str(e)
    return None
