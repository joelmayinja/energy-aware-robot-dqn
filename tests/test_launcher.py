import unittest
from unittest.mock import MagicMock, patch

from src import launcher


class DashboardLauncherTests(unittest.TestCase):
    def test_launcher_recognizes_the_combined_dashboard_health_endpoint(self):
        health_response = MagicMock()
        health_response.__enter__.return_value.read.return_value = (
            b'{"application":"energy-consumption-dqn"}'
        )
        page_response = MagicMock()
        page_response.__enter__.return_value.read.return_value = (
            b'<title>Wallie | Energy Intelligence</title><meta name="wattwise-entry" content="demo-auth-v1">'
        )

        with patch("src.launcher.urlopen", side_effect=[health_response, page_response]):
            self.assertEqual(launcher._backend_at_port(), "dashboard")

    def test_launcher_detects_the_old_separate_backend(self):
        health_response = MagicMock()
        health_response.__enter__.return_value.read.return_value = (
            b'{"application":"energy-consumption-dqn"}'
        )
        page_response = MagicMock()
        page_response.__enter__.return_value.read.return_value = (
            b'{"service":"api"}'
        )

        with patch("src.launcher.urlopen", side_effect=[health_response, page_response]):
            self.assertEqual(launcher._backend_at_port(), "legacy_dashboard")

    def test_launcher_rejects_an_unrelated_application_on_dashboard_port(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = (
            b'{"application":"another-service"}'
        )

        with patch("src.launcher.urlopen", return_value=response):
            self.assertEqual(launcher._backend_at_port(), "occupied")

    def test_launcher_opens_existing_combined_dashboard(self):
        with (
            patch("src.launcher._backend_at_port", return_value="dashboard"),
            patch("src.launcher.webbrowser.open") as open_browser,
        ):
            result = launcher.main()

        self.assertEqual(result, 0)
        open_browser.assert_called_once_with(launcher.URL)
