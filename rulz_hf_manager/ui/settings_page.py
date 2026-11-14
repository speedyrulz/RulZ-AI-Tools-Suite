"""Settings: Hugging Face token, download folder, appearance and server options."""

from __future__ import annotations

from nicegui import run, ui

from .. import __version__, config, hub
from ..platform_utils import open_folder, platform_label, to_local_path
from .common import frame, masked_hint, notify_error, notify_ok
from .pickers import pick_local

TOKEN_SOURCES = {
    'app': ('key', 'positive', 'Using the token saved in this app.'),
    'env': ('terminal', 'grey-6', 'Using the HF_TOKEN environment variable (no token saved in this app).'),
    'cli': ('terminal', 'grey-6', 'Using the token from your "hf auth login" (no token saved in this app).'),
    'none': ('key_off', 'warning', 'No token yet: you can only browse public repos.'),
}


@ui.page('/settings')
async def settings_page() -> None:
    s = config.get()

    with frame('settings') as look:
        ui.label('Settings').classes('rz-page-title')

        # ------------------------------------------------------------------ token
        with ui.card().classes('rulz-card w-full p-5 gap-3'):
            with ui.row().classes('items-center gap-3 no-wrap'):
                ui.icon('key').classes('rz-sec-icon')
                ui.label('Hugging Face access token').classes('rz-title')
            ui.markdown('Create one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). '
                        'Choose **Write** access (or a fine-grained token with write permissions on your repos) '
                        'so you can upload, move and delete files. The token is stored only on this computer '
                        f'in `{config.settings_path()}`.').classes('text-sm')
            source_row = ui.row().classes('items-center gap-2 text-sm')
            with ui.row().classes('w-full items-center gap-2 no-wrap'):
                # The saved token itself never goes to the browser (with "Whole network" anyone on the LAN could
                # read it from the page): the box shows a masked hint and takes a new token only.
                token_input = ui.input('Token', password=True, password_toggle_button=True) \
                    .props('outlined stack-label').classes('flex-1 rulz-mono')
                ui.button('Save & test', icon='save', on_click=lambda: save_token())
            with ui.row().classes('gap-2'):
                ui.button('Test current token', icon='verified_user', on_click=lambda: test_token()) \
                    .props('flat no-caps')
                import_btn = ui.button('Import from "hf auth login" / HF_TOKEN', icon='download',
                                       on_click=lambda: import_token()).props('flat no-caps')
                ui.button('Remove saved token', icon='delete', on_click=lambda: clear_token()) \
                    .props('flat no-caps color=negative')
            result = ui.column().classes('w-full gap-1')

        def render_source() -> None:
            token_input.props['placeholder'] = masked_hint(s.token, 'hf_...')
            token_input.update()
            src = hub.token_source()
            icon, color, text = TOKEN_SOURCES[src]
            source_row.clear()
            with source_row:
                ui.icon(icon, color=color)
                ui.label(text)
            import_btn.set_visibility(src in {'env', 'cli'} or (bool(hub.external_token()) and not s.token))

        async def test_token() -> None:
            result.clear()
            with result:
                ui.spinner()
            hub.forget_identity()
            try:
                ident = await run.io_bound(hub.whoami, True)
            except Exception as e:  # noqa: BLE001
                result.clear()
                with result, ui.row().classes('items-center gap-2 rz-err'):
                    ui.icon('error')
                    ui.label(hub.friendly_error(e))
                return
            result.clear()
            with result:
                if not ident:
                    ui.label('No token configured.').classes('rz-warn')
                    return
                with ui.row().classes('items-center gap-3'):
                    if ident.avatar:
                        ui.image(ident.avatar).classes('w-10 h-10 rounded-full')
                    with ui.column().classes('gap-0'):
                        ui.label(f'Signed in as {ident.fullname or ident.name} (@{ident.name})') \
                            .classes('font-semibold rz-ok')
                        if ident.is_oauth:
                            until = f', valid until {ident.expires:%d %b %Y}' if ident.expires else ''
                            ui.label(f'Temporary login token from "hf auth login" (OAuth){until}') \
                                .classes('text-sm rulz-muted')
                        else:
                            ui.label(f'Token "{ident.token_name or "unnamed"}", '
                                     f'permission: {ident.token_role or "?"}').classes('text-sm rulz-muted')
                        if ident.orgs:
                            ui.label('Organizations: ' + ', '.join(ident.orgs)).classes('text-sm rulz-muted')
                if ident.is_oauth:
                    with ui.row().classes('items-center gap-2 rz-warn no-wrap'):
                        ui.icon('schedule')
                        ui.label('This login token expires on its own. For a token that keeps working, create an '
                                 'access token with Write permission at huggingface.co/settings/tokens and save it '
                                 'here.')
                if ident.token_role == 'read':
                    with ui.row().classes('items-center gap-2 rz-warn'):
                        ui.icon('warning')
                        ui.label('This is a read-only token: browsing and downloading work, but uploads, moves, '
                                 'renames and deletes will fail.')

        async def store_token(value: str, note: str) -> None:
            s.token = value
            config.save()
            hub.forget_identity()
            token_input.value = ''
            render_source()
            notify_ok(note)
            await test_token()

        async def save_token() -> None:
            value = (token_input.value or '').strip()
            if not value:
                if s.token:  # an empty box keeps the saved token ("Remove saved token" removes it)
                    await test_token()
                else:
                    notify_error('Paste a token first.')
                return
            if not value.startswith('hf_'):
                ui.notify('Hugging Face tokens usually start with "hf_". Saved anyway.', type='warning')
            await store_token(value, 'Token saved')

        async def clear_token() -> None:
            await store_token('', 'Token removed')

        async def import_token() -> None:
            tok = hub.external_token()
            if not tok:
                notify_error('No token found from "hf auth login" or HF_TOKEN.')
                return
            await store_token(tok, 'Token imported')

        # ------------------------------------------------------------------ downloads
        with ui.card().classes('rulz-card w-full p-5 gap-3'):
            with ui.row().classes('items-center gap-3 no-wrap'):
                ui.icon('download').classes('rz-sec-icon')
                ui.label('Downloads').classes('rz-title')
            ui.label('"Download to my download folder" saves files here, keeping the repo folder structure '
                     '(<folder>/<type>s/<owner>/<name>/...).').classes('text-sm rulz-muted')
            with ui.row().classes('w-full items-center gap-2 no-wrap'):
                dl_input = ui.input('Download folder', value=s.download_dir,
                                    placeholder=str(s.effective_download_dir())) \
                    .props('outlined stack-label').classes('flex-1 rulz-mono')

                async def browse() -> None:
                    picked = await pick_local('folder', 'Choose the download folder', s.effective_download_dir()
                                              if s.effective_download_dir().exists() else None)
                    if picked:
                        dl_input.value = picked[0]
                        save_dl()
                ui.button(icon='folder_open', on_click=browse).props('flat round').tooltip('Browse')
                ui.button(icon='launch', on_click=lambda: open_dl()).props('flat round').tooltip('Open folder')

            def save_dl() -> None:
                new_value = to_local_path(dl_input.value or '')
                if new_value == s.download_dir:
                    return
                s.download_dir = new_value
                dl_input.value = s.download_dir
                config.save()
                notify_ok(f'Downloads go to {s.effective_download_dir()}')
            dl_input.on('blur', save_dl)
            dl_input.on('keydown.enter', save_dl)

            def open_dl() -> None:
                d = s.effective_download_dir()
                d.mkdir(parents=True, exist_ok=True)
                if not open_folder(d):
                    ui.notify(f'Folder: {d}')

        # ------------------------------------------------------------------ appearance & behaviour
        with ui.card().classes('rulz-card w-full p-5 gap-3'):
            with ui.row().classes('items-center gap-3 no-wrap'):
                ui.icon('tune').classes('rz-sec-icon')
                ui.label('Appearance & behaviour').classes('rz-title')
            with ui.row().classes('items-center gap-3'):
                ui.label('Look')
                look_toggle = ui.toggle({'dark': 'Dark', 'light': 'Light'}, value='light' if look.light else 'dark',
                                        on_change=lambda e: e.value and look.set(e.value == 'light')) \
                    .props('no-caps unelevated toggle-color=primary')
                ui.label('Also with the sun / moon button at the top right.').classes('text-xs rulz-muted')
            look.on_change(lambda light: look_toggle.set_value('light' if light else 'dark'))

            def set_flag(name: str, value) -> None:
                setattr(s, name, value)
                config.save()
            ui.switch('Show hidden files (names starting with ".")', value=s.show_hidden,
                      on_change=lambda e: set_flag('show_hidden', e.value))
            ui.switch('Show when repo files were last updated (filled in after the list; a few seconds for big '
                      'folders)', value=s.repo_dates, on_change=lambda e: set_flag('repo_dates', e.value))
            with ui.row().classes('items-center gap-3'):
                ui.label('Rows per page')
                ui.select([50, 100, 250, 0], value=s.page_size if s.page_size in (50, 100, 250, 0) else 100,
                          on_change=lambda e: set_flag('page_size', e.value)).props('outlined dense') \
                    .classes('w-28').tooltip('0 = all')

        # ------------------------------------------------------------------ server
        with ui.card().classes('rulz-card w-full p-5 gap-3'):
            with ui.row().classes('items-center gap-3 no-wrap'):
                ui.icon('dns').classes('rz-sec-icon')
                ui.label('Server').classes('rz-title')
            ui.label('Changes here apply the next time you start the app.').classes('text-sm rulz-muted')
            with ui.row().classes('items-center gap-3'):
                port = ui.number('Port', value=s.port, min=1024, max=65535, precision=0) \
                    .props('outlined dense').classes('w-32')
                host = ui.select({'127.0.0.1': 'This computer only (recommended)',
                                  '0.0.0.0': 'Whole network (anyone on your LAN can use your token!)'},
                                 value=s.host if s.host in ('127.0.0.1', '0.0.0.0') else '127.0.0.1',
                                 label='Listen on').props('outlined dense').classes('w-96')

                def save_server() -> None:
                    s.port = int(port.value or 8765)
                    s.host = host.value
                    config.save()
                    notify_ok('Saved. Restart the app to apply.')
                ui.button('Save', icon='save', on_click=save_server).props('flat no-caps')
            with ui.column().classes('gap-0 text-sm rulz-muted'):
                ui.label(f'RulZ Huggingface Manager {__version__}  /  running on {platform_label()}')
                ui.label(f'Settings file: {config.settings_path()}').classes('rulz-mono')
                try:
                    import huggingface_hub
                    import nicegui
                    ui.label(f'huggingface_hub {huggingface_hub.__version__}  /  NiceGUI {nicegui.__version__}')
                except Exception:  # noqa: BLE001
                    pass

    render_source()
    if hub.token_source() != 'none':
        ui.timer(0.05, test_token, once=True)
