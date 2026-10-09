from __future__ import annotations

import json

from bentotruck.edamame import ConversationMemory, LongTermMemory, VectorMemory, WorkingMemory


class TestConversationMemory:
    def test_recall_returns_recent_items_in_order(self):
        memory = ConversationMemory(max_items=10)
        memory.add("user", "first")
        memory.add("assistant", "second")
        memory.add("user", "third")

        recalled = memory.recall(k=2)
        assert [item.content for item in recalled] == ["second", "third"]

    def test_rolling_window_drops_oldest(self):
        memory = ConversationMemory(max_items=2)
        memory.add("user", "one")
        memory.add("user", "two")
        memory.add("user", "three")

        recalled = memory.recall(k=10)
        assert [item.content for item in recalled] == ["two", "three"]

    def test_clear(self):
        memory = ConversationMemory()
        memory.add("user", "hi")
        memory.clear()
        assert memory.recall() == []


class TestWorkingMemory:
    def test_set_and_get_by_default_key(self):
        memory = WorkingMemory()
        memory.add("username", "alice")
        assert memory.get("username") == "alice"
        assert memory.get("missing", "default") == "default"

    def test_explicit_key_overrides_role(self):
        memory = WorkingMemory()
        memory.add("assistant", "42", key="answer")
        assert memory.get("answer") == "42"
        assert memory.get("assistant") is None

    def test_clear(self):
        memory = WorkingMemory()
        memory.add("k", "v")
        memory.clear()
        assert memory.get("k") is None


class TestVectorMemory:
    def test_recall_without_query_returns_recent(self):
        memory = VectorMemory()
        memory.add("user", "apple banana")
        memory.add("user", "car truck")
        recalled = memory.recall(k=1)
        assert recalled[0].content == "car truck"

    def test_recall_ranks_by_similarity(self):
        memory = VectorMemory()
        memory.add("user", "the cat sat on the mat")
        memory.add("user", "rockets launch into orbit")
        memory.add("user", "cats and dogs are pets")

        recalled = memory.recall("tell me about cats", k=1)
        assert "cat" in recalled[0].content

    def test_custom_embed_fn(self):
        calls = []

        def embed_fn(text):
            calls.append(text)
            return [1.0, 0.0] if "match" in text else [0.0, 1.0]

        memory = VectorMemory(embed_fn=embed_fn)
        memory.add("user", "this will match")
        memory.add("user", "this will not")

        recalled = memory.recall("match please", k=1)
        assert recalled[0].content == "this will match"
        assert len(calls) == 3  # two adds + one query


class TestLongTermMemory:
    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "memory.json"

        memory = LongTermMemory(path)
        memory.add("user", "remember this")

        assert path.exists()
        data = json.loads(path.read_text())
        assert data[0]["content"] == "remember this"

        reloaded = LongTermMemory(path)
        recalled = reloaded.recall(k=5)
        assert [item.content for item in recalled] == ["remember this"]

    def test_clear_persists_empty_state(self, tmp_path):
        path = tmp_path / "memory.json"
        memory = LongTermMemory(path)
        memory.add("user", "temp")
        memory.clear()

        reloaded = LongTermMemory(path)
        assert reloaded.recall() == []


def test_graph_memory_parses_facts_and_recalls_neighbors():
    from bentotruck.edamame import Fact, GraphMemory

    memory = GraphMemory(depth=1)
    memory.add("user", "Ada works at Bento Truck.")
    memory.add("user", "Bento Truck is located in Austin")
    memory.add_fact("Austin", "is in", "Texas")
    memory.add("user", "Grace likes ramen", subject="Grace", relation="likes", object="ramen")
    memory.add("note", "we met Linus at the Market today")  # not a pattern: linked by entity

    assert memory.facts(subject="ada") == [Fact("Ada", "works at", "Bento Truck")]
    assert memory.facts(subject="Bento Truck") == [Fact("Bento Truck", "is located in", "Austin")]
    recalled = [item.content for item in memory.recall("Where does Ada work?", k=10)]
    assert recalled == ["Ada works at Bento Truck", "Bento Truck is located in Austin"]
    deeper = GraphMemory(depth=2)
    for fact in memory.facts():
        deeper.add_fact(fact.subject, fact.relation, fact.object)
    assert "Austin is in Texas" in [i.content for i in deeper.recall("Ada", k=10)]
    assert memory.facts(subject="Linus")[0].relation == "mentioned in"
    assert [i.content for i in memory.recall(None, k=1)] == [str(memory.facts()[-1])]
    memory.clear()
    assert memory.recall("Ada") == []
