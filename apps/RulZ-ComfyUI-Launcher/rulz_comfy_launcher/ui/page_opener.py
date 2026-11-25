"""Open browser pages after a non-blocking RulZFlow availability check."""
from PySide6.QtCore import QObject, QTimer, QUrl
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkProxy, QNetworkRequest

from ..command import page_url


class PageOpener(QObject):
    def __init__(self, open_url, status, parent=None):
        super().__init__(parent)
        self._open_url = open_url
        self._status = status
        self._pending = {}
        self._stopped = False
        self._network = QNetworkAccessManager(self)
        self._network.setProxy(QNetworkProxy(QNetworkProxy.ProxyType.NoProxy))

    def open(self, profile_id, name, server_url, rulzflow, valid):
        if self._stopped or profile_id in self._pending or not valid():
            return
        if not rulzflow:
            self._open_url(QUrl(server_url))
            return
        target = page_url(server_url, True)
        request = QNetworkRequest(QUrl(target))
        request.setAttribute(QNetworkRequest.Attribute.RedirectPolicyAttribute,
                             QNetworkRequest.RedirectPolicy.ManualRedirectPolicy)
        reply = self._network.get(request)
        self._pending[profile_id] = reply
        timer = QTimer(reply)
        timer.setSingleShot(True)
        timer.timeout.connect(reply.abort)
        timer.start(3000)

        def finished():
            self._pending.pop(profile_id, None)
            code = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
            reply.deleteLater()
            if self._stopped or not valid():
                return
            if code in (404, 503):
                reason = "is not installed" if code == 404 else "has missing page files"
                self._status(f"RulZFlow {reason} on {name}: opened the normal ComfyUI page.")
                self._open_url(QUrl(server_url))
            else:
                self._open_url(QUrl(target))

        reply.finished.connect(finished)

    def stop(self):
        self._stopped = True
        for reply in list(self._pending.values()):
            reply.abort()
