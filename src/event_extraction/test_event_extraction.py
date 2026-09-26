import sys
import unittest
from pathlib import Path

# Ensure the event_extraction package directory is on sys.path.
PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from event_extractor import (
    compute_confidence,
    detect_entities,
    detect_locations,
    deduplicate_events,
    find_keywords_in_text,
    article_to_events,
)
from event_models import EVENT_KEYWORDS


class EventExtractionTests(unittest.TestCase):
    def test_location_detection_chennai(self):
        locations = detect_locations("Heavy rain in Chennai disrupted logistics.")
        self.assertIn("Chennai", locations)

    def test_location_detection_bengaluru_bangalore(self):
        locations = detect_locations("A factory shutdown near Bangalore affected production.")
        self.assertIn("Bengaluru", locations)

    def test_location_detection_state(self):
        locations = detect_locations("A major logistics strike in Karnataka slowed shipments.")
        self.assertIn("Karnataka", locations)

    def test_affected_entity_tata_electronics(self):
        entities = detect_entities("Tata Electronics reported a component shortage.")
        self.assertIn("Tata Electronics", entities)

    def test_affected_entity_dixon_technologies(self):
        entities = detect_entities("Dixon Technologies and its supplier network were impacted.")
        self.assertIn("Dixon Technologies", entities)

    def test_heavy_rain_detection(self):
        events = article_to_events(
            {
                "title": "Heavy rain shuts down ports in Chennai",
                "description": "Port congestion followed heavy rain.",
                "publishedAt": "2026-08-12T08:00:00Z",
                "source": {"name": "Test News"},
            },
            Path("dummy_news.json"),
        )
        self.assertTrue(any(ev["Event_Type"] == "Heavy Rain" for ev in events))

    def test_port_congestion_detection(self):
        events = article_to_events(
            {
                "title": "Port congestion delays laptop shipments",
                "description": "Shipments are delayed as docks fill up.",
                "publishedAt": "2026-08-12T09:00:00Z",
                "source": {"name": "Trade News"},
            },
            Path("dummy_news.json"),
        )
        self.assertTrue(any(ev["Event_Type"] == "Port Congestion" for ev in events))

    def test_supply_shortage_detection(self):
        events = article_to_events(
            {
                "title": "Component shortage hits smartphone production",
                "description": "Suppliers struggle to meet demand.",
                "publishedAt": "2026-08-12T10:00:00Z",
                "source": {"name": "Market News"},
            },
            Path("dummy_news.json"),
        )
        self.assertTrue(any(ev["Event_Type"] == "Component Shortage" for ev in events))

    def test_unknown_location_handling(self):
        events = article_to_events(
            {
                "title": "Supply shortage affects electronics firms",
                "description": "Manufacturers are still waiting for components.",
                "publishedAt": "2026-08-12T11:00:00Z",
                "source": {"name": "Supply Chain News"},
            },
            Path("dummy_news.json"),
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["Location"], "Unknown")

    def test_unknown_entity_handling(self):
        events = article_to_events(
            {
                "title": "Power outage shuts down factory",
                "description": "The outage affected manufacturing lines.",
                "publishedAt": "2026-08-12T12:00:00Z",
                "source": {"name": "Factory News"},
            },
            Path("dummy_news.json"),
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["Affected_Entity"], "Unknown")

    def test_confidence_score_range(self):
        score = compute_confidence(
            event_type="Heavy Rain",
            title="Heavy rain warning in Chennai",
            description="A severe weather system is expected to cause flooding.",
            location_found=True,
            entity_found=False,
            keyword_matches=2,
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_deduplication(self):
        events = [
            {
                "Event_Type": "Heavy Rain",
                "Date": "2026-08-12T08:00:00Z",
                "Location": "Chennai",
                "Title": "Heavy Rain in Chennai",
                "Affected_Entity": "Unknown",
                "Source": "dummy_news.json",
                "Description": "Heavy rain is forecast.",
            },
            {
                "Event_Type": "Heavy Rain",
                "Date": "2026-08-12T08:00:00Z",
                "Location": "Chennai",
                "Title": "Heavy Rain in Chennai",
                "Affected_Entity": "Unknown",
                "Source": "dummy_news.json",
                "Description": "Heavy rain is forecast.",
            },
        ]
        unique = deduplicate_events(events)
        self.assertEqual(len(unique), 1)

    def test_find_keywords_in_text(self):
        matches = find_keywords_in_text("Port congestion and shipping delay reported.", EVENT_KEYWORDS)
        self.assertIn("Port Congestion", matches)
        self.assertIn("Shipping Delay", matches)


if __name__ == "__main__":
    unittest.main()
