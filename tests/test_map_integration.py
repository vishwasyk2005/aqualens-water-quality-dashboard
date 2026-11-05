"""Contract checks for local map dependencies and restricted external tile images."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server import Handler


class MapIntegrationTests(unittest.TestCase):
    def test_map_dependencies_are_local_and_loaded_in_order(self):
        page = (ROOT / 'web' / 'index.html').read_text(encoding='utf-8-sig')
        self.assertLess(page.index('vendor/leaflet/leaflet.css'), page.index('href="styles.css"'))
        self.assertLess(page.index('src="vendor/leaflet/leaflet.js"'), page.index('src="water-map.js"'))
        self.assertLess(page.index('src="water-map.js"'), page.index('src="app.js"'))
        self.assertNotIn('src="https://', page)
        self.assertEqual(page.count('data-page="map"'), 1)
        self.assertIn('id="page-select" aria-label="Dashboard page"', page)
        self.assertIn('<option value="map">', page)

    def test_tile_policy_does_not_allow_remote_scripts_or_data_connections(self):
        handler = Handler.__new__(Handler)
        with patch.object(handler, 'send_response'), patch.object(handler, 'send_header') as header, patch.object(handler, 'end_headers'):
            handler.response_headers(200, 'text/html', 100)
        headers = dict(call.args for call in header.call_args_list)
        policy = dict(part.strip().split(' ', 1) for part in headers['Content-Security-Policy'].split(';') if part.strip())
        self.assertEqual(policy['script-src'], "'self'")
        self.assertEqual(policy['connect-src'], "'self'")
        self.assertEqual(policy['img-src'].split(), ["'self'", 'data:', 'https://tile.openstreetmap.org', 'https://server.arcgisonline.com'])
        self.assertEqual(headers['Referrer-Policy'], 'strict-origin-when-cross-origin')


if __name__ == '__main__':
    unittest.main()
