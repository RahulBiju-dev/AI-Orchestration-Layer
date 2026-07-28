import json
import unittest

from agent.cancellation import CancellationToken, OperationCancelled
from tools.knowledge_graph_builder import knowledge_graph_builder

class TestKnowledgeGraphBuilder(unittest.TestCase):
    def test_knowledge_graph_builder_concept_limit(self):
        # Create 501 concepts
        concepts = [{"id": f"c{i}", "label": f"Concept {i}"} for i in range(501)]
        relationships = []

        result = knowledge_graph_builder(concepts, relationships)
        expected_json = json.dumps({"error": "Graph exceeds the 500-concept/3000-relationship safety limit"})

        self.assertEqual(result, expected_json)

    def test_knowledge_graph_builder_relationship_limit(self):
        # Create 3001 relationships
        concepts = []
        relationships = [{"source": "c1", "target": "c2", "type": "related_to"} for _ in range(3001)]

        result = knowledge_graph_builder(concepts, relationships)
        expected_json = json.dumps({"error": "Graph exceeds the 500-concept/3000-relationship safety limit"})

        self.assertEqual(result, expected_json)

    def test_domain_specific_edges_are_traversed_without_an_explicit_filter(self):
        payload = json.loads(knowledge_graph_builder(
            [{"id": "api"}, {"id": "service"}, {"id": "database"}],
            [
                {
                    "id": "r1",
                    "source": "api",
                    "target": "service",
                    "type": "depends_on",
                    "weight": 0.9,
                },
                {
                    "id": "r2",
                    "source": "service",
                    "target": "database",
                    "type": "uses",
                    "weight": 0.8,
                },
            ],
            query={"source": "api", "target": "database"},
        ))

        self.assertNotIn("error", payload)
        path = payload["analysis"]["inferred_paths"][0]
        self.assertEqual(path["target"], "database")
        self.assertEqual(path["inferred_effect"], "unspecified")
        self.assertEqual(path["confidence"], 0.8)

    def test_direct_query_match_is_not_omitted(self):
        payload = json.loads(knowledge_graph_builder(
            [{"id": "a"}, {"id": "b"}],
            [{
                "id": "direct",
                "source": "a",
                "target": "b",
                "type": "causes",
                "evidence": ["source-1"],
            }],
            query={"source": "a", "target": "b"},
        ))

        self.assertEqual(
            payload["analysis"]["direct_matches"],
            [{
                "edge": "direct",
                "from": "a",
                "type": "causes",
                "to": "b",
                "weight": 1.0,
                "evidence": ["source-1"],
            }],
        )

    def test_feedback_components_are_complete_and_edge_backed(self):
        payload = json.loads(knowledge_graph_builder(
            [{"id": value} for value in ("a", "b", "c", "outside")],
            [
                {"source": "a", "target": "b", "type": "depends_on"},
                {"source": "b", "target": "a", "type": "depends_on"},
                {"source": "b", "target": "c", "type": "depends_on"},
                {"source": "c", "target": "b", "type": "depends_on"},
                {"source": "outside", "target": "a", "type": "uses"},
            ],
        ))

        self.assertEqual(
            payload["analysis"]["feedback_components"],
            [{
                "concepts": ["a", "b", "c"],
                "concept_count": 3,
                "relationship_count": 4,
            }],
        )
        cycle = payload["analysis"]["potential_feedback_cycles"][0]
        edges = {
            (relation["source"], relation["target"])
            for relation in payload["graph"]["relationships"]
        }
        self.assertTrue(all(
            (source, target) in edges
            for source, target in zip(cycle, cycle[1:])
        ))

    def test_only_explicit_positive_and_negative_effects_conflict(self):
        payload = json.loads(knowledge_graph_builder(
            [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            [
                {"source": "a", "target": "b", "type": "related_to"},
                {"source": "a", "target": "b", "type": "inhibits"},
                {"source": "a", "target": "c", "type": "supports"},
                {"source": "a", "target": "c", "type": "inhibits"},
            ],
        ))

        self.assertEqual(payload["analysis"]["contradictions"], [{
            "source": "a",
            "target": "c",
            "positive": ["supports"],
            "negative": ["inhibits"],
        }])

    def test_invalid_values_are_rejected_instead_of_coerced_or_truncated(self):
        too_long = "x" * 2001
        payload = json.loads(knowledge_graph_builder(
            [{"id": "a", "label": too_long}],
            [],
        ))
        self.assertIn("label exceeds", " ".join(payload["details"]))

        payload = json.loads(knowledge_graph_builder(
            [{"id": "a"}],
            [],
            max_depth=99,
        ))
        self.assertIn("between 1 and 8", payload["error"])

        payload = json.loads(knowledge_graph_builder(
            [{"id": "a"}, {"id": "b"}],
            [{"source": "a", "target": "b", "type": "uses", "weight": True}],
        ))
        self.assertIn("weight must be numeric", " ".join(payload["details"]))

    def test_large_graph_echo_is_truthfully_bounded(self):
        concepts = [
            {"id": f"c{index}", "label": f"Concept {index} " + "x" * 1000}
            for index in range(50)
        ]
        payload = json.loads(knowledge_graph_builder(concepts, []))

        self.assertTrue(payload["graph"]["echo_truncated"])
        self.assertGreater(payload["graph"]["omitted_concepts"], 0)
        self.assertEqual(payload["graph"]["concept_count"], 50)

    def test_large_analysis_remains_parseable_inside_tool_output_limit(self):
        concepts = [
            {"id": f"node-{index}-" + "x" * 120}
            for index in range(100)
        ]
        relationships = [
            {
                "source": concepts[index]["id"],
                "target": concepts[(index + 1) % len(concepts)]["id"],
                "type": "depends_on",
            }
            for index in range(len(concepts))
        ]
        raw = knowledge_graph_builder(concepts, relationships)
        payload = json.loads(raw)

        self.assertLess(len(raw), 100_000)
        self.assertTrue(payload["graph"]["echo_truncated"])
        self.assertTrue(payload["analysis"]["traversal_truncated"])
        self.assertTrue(payload["analysis"]["feedback_components_truncated"])

    def test_cancellation_is_honored_before_graph_work(self):
        token = CancellationToken()
        token.cancel("stop graph")
        with self.assertRaises(OperationCancelled):
            knowledge_graph_builder(
                [{"id": "a"}],
                [],
                cancellation_token=token,
            )

if __name__ == '__main__':
    unittest.main()
