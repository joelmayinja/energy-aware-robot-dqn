from http.server import ThreadingHTTPServer
import threading
import unittest
from unittest.mock import MagicMock, patch
from urllib.request import urlopen

from src import launcher
from src.launcher import FRONTEND_ROOT, FrontendRequestHandler


class FrontendServerTests(unittest.TestCase):
    def test_frontend_root_serves_renamed_web_entrypoint(self):
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0), launcher.FrontendRequestHandler
        )
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            with urlopen(f"http://127.0.0.1:{server.server_port}/", timeout=2) as response:
                page = response.read().decode("utf-8")

            self.assertIn("<title>Wallie | Energy Intelligence</title>", page)
            self.assertTrue((FRONTEND_ROOT / "web.html").is_file())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_frontend_serves_assets_from_renamed_directory(self):
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0), launcher.FrontendRequestHandler
        )
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            with urlopen(
                f"http://127.0.0.1:{server.server_port}/assets/.gitkeep", timeout=2
            ) as response:
                status = response.status

            self.assertEqual(status, 200)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_launcher_recognizes_the_renamed_frontend_page(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = (
            b"<title>Wallie | Energy Intelligence</title>"
        )

        with patch("src.launcher.urlopen", return_value=response):
            self.assertEqual(launcher._frontend_at_port(), "frontend")

    def test_launcher_does_not_claim_an_unrelated_page(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b"<title>Other site</title>"

        with patch("src.launcher.urlopen", return_value=response):
            self.assertEqual(launcher._frontend_at_port(), "occupied")
