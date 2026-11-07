"""'My Hub' page: every model, dataset, space and bucket in a namespace at a glance."""

from __future__ import annotations

from nicegui import run, ui

from .. import config, hub
from .common import SIZE_FORMAT_JS, browse_url, fmt_ago, fmt_date, frame, notify_error
from .repo_actions import create_dialog, delete_dialog, duplicate_dialog, rename_dialog, visibility_dialog

_last_namespace: str | None = None


@ui.page('/')
async def dashboard_page() -> None:
    settings = config.get()
    state: dict = {'rows': [], 'namespaces': [], 'ident': None}

    with frame('hub'):
        welcome = ui.column().classes('w-full')
        with ui.card().classes('rulz-card w-full p-4 gap-3'):
            with ui.row().classes('w-full items-center gap-3'):
                ns_select = ui.select([], label='Namespace', with_input=True, new_value_mode='add-unique') \
                    .props('outlined dense').classes('w-64')
                ns_select.tooltip('Your account and organizations. Type any username to browse their public repos.')
                kind_toggle = ui.toggle({'all': 'All', **{k: hub.KIND_LABEL[k] + 's' for k in hub.KINDS}},
                                        value='all').props('no-caps rounded unelevated toggle-color=primary')
                search = ui.input(placeholder='Filter by name...').props('outlined dense clearable').classes('w-60')
                with search.add_slot('prepend'):
                    ui.icon('filter_list')
                ui.space()
                ui.button(icon='refresh', on_click=lambda: load()).props('flat round').tooltip('Refresh')
                new_btn = ui.button('New', icon='add', on_click=lambda: create_dialog(
                    state['namespaces'], ns_select.value or '', kind_toggle.value if kind_toggle.value != 'all' else 'model'))
            stats = ui.row().classes('w-full gap-3')

        shortcuts = ui.row().classes('w-full gap-2 items-center')

        columns = [
            {'name': 'name', 'label': 'Name', 'field': 'id', 'align': 'left', 'sortable': True},
            {'name': 'kind', 'label': 'Type', 'field': 'kind_label', 'align': 'left', 'sortable': True},
            {'name': 'visibility', 'label': 'Visibility', 'field': 'visibility', 'align': 'left', 'sortable': True},
            {'name': 'size', 'label': 'Size', 'field': 'size', 'align': 'right', 'sortable': True,
             ':format': SIZE_FORMAT_JS},
            {'name': 'updated', 'label': 'Updated', 'field': 'updated', 'align': 'left', 'sortable': True},
            {'name': 'stats', 'label': '', 'field': 'stats', 'align': 'left'},
            {'name': 'actions', 'label': '', 'field': 'id', 'align': 'right'},
        ]
        table = ui.table(columns=columns, rows=[], row_key='key',
                         pagination={'rowsPerPage': 50, 'sortBy': 'updated', 'descending': True}) \
            .props('flat bordered binary-state-sort :rows-per-page-options="[25,50,100,0]" '
                   'no-data-label="Nothing here yet." loading-label="Loading..." no-results-label="No match."') \
            .classes('w-full rulz-table')
        table.bind_filter_from(search, 'value')
        table.add_slot('body-cell-name', r'''
            <q-td :props="props">
              <div class="row items-center no-wrap q-gutter-x-sm cursor-pointer rulz-name"
                   @click="$parent.$emit('browse', props.row)">
                <q-icon :name="props.row.icon" class="rz-kind-icon" size="22px"/>
                <span class="text-weight-medium">{{ props.row.id }}</span>
                <q-icon v-if="props.row.favorite" name="star" class="rz-kind-icon" size="16px"/>
              </div>
            </q-td>''')
        table.add_slot('body-cell-visibility', r'''
            <q-td :props="props">
              <span class="rz-vis" :class="props.row.visibility">
                <q-icon :name="props.row.visibility === 'public' ? 'public' : 'lock'"/>{{ props.row.visibility }}
              </span>
            </q-td>''')
        table.add_slot('body-cell-updated', r'''
            <q-td :props="props"><span :title="props.row.updated">{{ props.row.updated_ago }}</span></q-td>''')
        table.add_slot('body-cell-actions', r'''
            <q-td :props="props" auto-width>
              <q-btn flat dense round size="sm" icon="folder_open" @click.stop="$parent.$emit('browse', props.row)">
                <q-tooltip>Browse files</q-tooltip></q-btn>
              <q-btn flat dense round size="sm" icon="open_in_new" @click.stop="$parent.$emit('web', props.row)">
                <q-tooltip>Open on huggingface.co</q-tooltip></q-btn>
              <q-btn flat dense round size="sm" icon="more_vert" @click.stop>
                <q-menu auto-close>
                  <q-list dense style="min-width: 210px">
                    <q-item clickable @click="$parent.$emit('act', {action: 'favorite', row: props.row})">
                      <q-item-section avatar><q-icon :name="props.row.favorite ? 'star' : 'star_outline'"/></q-item-section>
                      <q-item-section>{{ props.row.favorite ? 'Remove from favorites' : 'Add to favorites' }}</q-item-section></q-item>
                    <q-item clickable @click="$parent.$emit('act', {action: 'rename', row: props.row})">
                      <q-item-section avatar><q-icon name="drive_file_rename_outline"/></q-item-section>
                      <q-item-section>Rename / transfer</q-item-section></q-item>
                    <q-item clickable @click="$parent.$emit('act', {action: 'visibility', row: props.row})">
                      <q-item-section avatar><q-icon :name="props.row.visibility === 'public' ? 'lock' : 'public'"/></q-item-section>
                      <q-item-section>{{ props.row.visibility === 'public' ? 'Make private' : 'Make public' }}</q-item-section></q-item>
                    <q-item v-if="props.row.kind !== 'bucket'" clickable @click="$parent.$emit('act', {action: 'duplicate', row: props.row})">
                      <q-item-section avatar><q-icon name="content_copy"/></q-item-section>
                      <q-item-section>Duplicate</q-item-section></q-item>
                    <q-separator/>
                    <q-item clickable class="text-negative" @click="$parent.$emit('act', {action: 'delete', row: props.row})">
                      <q-item-section avatar><q-icon name="delete_forever" color="negative"/></q-item-section>
                      <q-item-section>Delete...</q-item-section></q-item>
                  </q-list>
                </q-menu>
              </q-btn>
            </q-td>''')

    def to_row(r: hub.RepoRow) -> dict:
        bits = []
        if r.downloads is not None:
            bits.append(f'{r.downloads:,} downloads')
        if r.likes:
            bits.append(f'{r.likes} likes')
        if r.files is not None:
            bits.append(f'{r.files:,} files')
        return {
            'key': f'{r.kind}:{r.id}', 'id': r.id, 'kind': r.kind, 'kind_label': hub.KIND_LABEL[r.kind],
            'icon': hub.KIND_ICON[r.kind], 'visibility': r.visibility,
            'size': r.size, 'updated': fmt_date(r.updated), 'updated_ago': fmt_ago(r.updated),
            'stats': ' / '.join(bits), 'favorite': settings.is_favorite(r.kind, r.id),
        }

    def apply_filter() -> None:
        kind = kind_toggle.value
        rows = [to_row(r) for r in state['rows'] if kind == 'all' or r.kind == kind]
        table.update_rows(rows)
        stats.clear()
        with stats:
            for k in hub.KINDS:
                items = [r for r in state['rows'] if r.kind == k]
                total = sum(r.size or 0 for r in items)
                # Click a tile to show only that type, click it again to show everything.
                with ui.row().classes('rulz-stat items-center gap-3 no-wrap' + (' rz-on' if kind == k else '')) \
                        .on('click', lambda k=k: kind_toggle.set_value('all' if kind_toggle.value == k else k)):
                    ui.icon(hub.KIND_ICON[k]).classes('rz-sec-icon')
                    with ui.column().classes('gap-0'):
                        ui.label(f'{len(items)} {hub.KIND_LABEL[k]}{"s" if len(items) != 1 else ""}') \
                            .classes('rz-stat-value')
                        ui.label(hub.human_size(total) if total else '-').classes('rz-stat-sub')

    kind_toggle.on_value_change(apply_filter)

    def render_shortcuts() -> None:
        shortcuts.clear()
        favs, recent = settings.favorites, [r for r in settings.recent if r not in settings.favorites][:8]
        if not favs and not recent:
            return
        with shortcuts:
            for label, icon, items in (('Favorites', 'star', favs), ('Recent', 'history', recent)):
                if not items:
                    continue
                ui.icon(icon).classes('ml-1 ' + ('rz-kind-icon' if icon == 'star' else 'rz-ic-muted'))
                ui.label(label).classes('text-xs uppercase rulz-muted mr-1')
                for it in items:
                    if it.get('kind') not in hub.KIND_ICON:
                        continue
                    ui.chip(it['id'], icon=hub.KIND_ICON[it['kind']],
                            on_click=lambda it=it: ui.navigate.to(browse_url(it['kind'], it['id']))) \
                        .props('clickable outline dense').classes('text-sm rz-shortcut')

    async def load() -> None:
        global _last_namespace
        ns = (ns_select.value or '').strip()
        if not ns:
            return
        _last_namespace = ns
        table.props('loading')
        try:
            state['rows'] = await run.io_bound(hub.list_namespace, ns)
        except Exception as e:  # noqa: BLE001
            notify_error(e)
            state['rows'] = []
        finally:
            table.props(remove='loading')
        apply_filter()

    ns_select.on_value_change(lambda: load())

    async def on_act(e) -> None:
        action, row = e.args['action'], e.args['row']
        kind, repo_id = row['kind'], row['id']
        if action == 'favorite':
            added = settings.toggle_favorite(kind, repo_id)
            ui.notify(f'{"Added to" if added else "Removed from"} favorites')
            render_shortcuts()
            apply_filter()
        elif action == 'rename':
            await rename_dialog(kind, repo_id, on_done=lambda _: load())
        elif action == 'visibility':
            await visibility_dialog(kind, repo_id, row['visibility'] != 'public', on_done=lambda _: load())
        elif action == 'duplicate':
            await duplicate_dialog(kind, repo_id)
        elif action == 'delete':
            await delete_dialog(kind, repo_id, on_done=load)

    table.on('browse', lambda e: ui.navigate.to(browse_url(e.args['kind'], e.args['id'])))
    table.on('web', lambda e: ui.navigate.to(hub.web_url(e.args['kind'], e.args['id']), new_tab=True))
    table.on('act', on_act)

    async def init() -> None:
        render_shortcuts()
        try:
            ident = await run.io_bound(hub.whoami)
        except Exception as e:  # noqa: BLE001
            ident = None
            notify_error(e, prefix='Could not verify your token: ')
        state['ident'] = ident
        if ident:
            state['namespaces'] = ident.namespaces
            ns_select.set_options(ident.namespaces)
        else:
            new_btn.disable()
            with welcome:
                with ui.card().classes('rulz-card rz-welcome w-full p-6 gap-2'):
                    with ui.row().classes('items-center gap-3 no-wrap'):
                        ui.icon('waving_hand').classes('rz-sec-icon')
                        ui.label('Welcome to RulZ Huggingface Manager').classes('rz-page-title')
                    ui.label('Add your Hugging Face access token to see and manage your own models, datasets, '
                             'spaces and buckets. Without a token you can still browse public repos: type a '
                             'username in "Namespace" or paste any repo link in the search box at the top.')
                    with ui.row().classes('gap-2 mt-2'):
                        ui.button('Add my token', icon='key', on_click=lambda: ui.navigate.to('/settings'))
                        ui.button('Get a token on huggingface.co', icon='open_in_new',
                                  on_click=lambda: ui.navigate.to('https://huggingface.co/settings/tokens',
                                                                  new_tab=True)).props('flat')
        default_ns = _last_namespace if _last_namespace else (ident.name if ident else '')
        if default_ns:
            if default_ns not in ns_select.options:
                ns_select.set_options([*ns_select.options, default_ns])
            ns_select.set_value(default_ns)  # triggers load()
        else:
            apply_filter()

    ui.timer(0.01, init, once=True)
