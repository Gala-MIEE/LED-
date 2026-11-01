"""Проверки нарезки, чтения Word, кеша и уточняющих вопросов. Сеть не нужна."""

import sqlite3
import sys
import tempfile
import unittest
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cache import RAGCache
from telegram_bot import (
    advance_offset,
    answer_for_telegram,
    remember_turn,
    split_telegram_message,
)
from rag_pipeline import NO_INFO_ANSWER, build_prompt, is_followup, retrieval_query
from vector_store import (
    CHUNKER_VERSION,
    collection_signature,
    chunk_knowledge_text,
    filter_by_distance,
    format_search_hits,
    project_path,
    read_text_document,
    resolve_data_file,
)

_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _sql(path, statement, params=()):
    connection = sqlite3.connect(path)
    try:
        connection.execute(statement, params)
        connection.commit()
    finally:
        connection.close()


def _docx(path: Path):
    document = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{_NS}">
  <w:body>
    <w:p>
      <w:pPr><w:pStyle w:val="H1"/></w:pPr>
      <w:r><w:t>Питание ленты</w:t></w:r>
    </w:p>
    <w:p>
      <w:r><w:t>## 1. Можно ли резать ленту в любом месте?</w:t></w:r>
    </w:p>
    <w:p>
      <w:r><w:t>Нет. Резать можно только по специальным меткам.</w:t></w:r>
    </w:p>
    <w:tbl>
      <w:tr>
        <w:tc><w:p><w:r><w:t>Зона</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>IP</w:t></w:r></w:p></w:tc>
      </w:tr>
      <w:tr>
        <w:tc><w:p><w:r><w:t>Ванная</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>IP65</w:t></w:r></w:p></w:tc>
      </w:tr>
    </w:tbl>
  </w:body>
</w:document>
"""
    styles = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{_NS}">
  <w:style w:type="paragraph" w:styleId="H1">
    <w:name w:val="heading 1"/>
  </w:style>
</w:styles>
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document.encode("utf-8"))
        archive.writestr("word/styles.xml", styles.encode("utf-8"))


class DocumentTests(unittest.TestCase):
    def test_docx_keeps_heading_table_and_markdown(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "base.docx"
            _docx(path)
            text = read_text_document(str(path))

        self.assertIn("# Питание ленты", text)
        self.assertIn("## 1. Можно ли резать ленту в любом месте?", text)
        self.assertNotIn("## ##", text)
        self.assertIn("Ванная | IP65", text)
        self.assertEqual(text.count("Ванная"), 1)

        chunks = chunk_knowledge_text(text)
        matched = [chunk for chunk in chunks if "резать" in chunk["text"]]
        self.assertTrue(matched)
        self.assertIn("меткам", matched[0]["text"])
        self.assertIn("IP65", matched[0]["text"])
        self.assertIn("резать", matched[0]["section"])

    def test_binary_doc_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old.doc"
            path.write_bytes(b"\xd0\xcf\x11\xe0")
            with self.assertRaises(ValueError):
                read_text_document(str(path))

    def test_lock_file_is_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "~$lock.docx").write_text("lock", encoding="utf-8")
            real = root / "base.docx"
            real.write_text("x", encoding="utf-8")
            self.assertEqual(Path(resolve_data_file(str(root))), real)

    def test_question_stays_with_answer(self):
        text = (
            "# База\n\n"
            "## 1. Можно ли резать ленту?\n\n"
            "Нет. Резать можно только по меткам, каждые 5 см для ленты 12 В.\n\n"
            "## 2. Какая гарантия на ленты?\n\n"
            "Гарантия на ленты и модули составляет 24 месяца, на блоки питания 12 месяцев.\n"
        )
        chunks = chunk_knowledge_text(text)
        cutting = next(chunk for chunk in chunks if "резать" in chunk["text"])
        warranty = next(chunk for chunk in chunks if "гарантия" in chunk["text"].lower())
        self.assertIn("меткам", cutting["text"])
        self.assertNotIn("24 месяца", cutting["text"])
        self.assertIn("24 месяца", warranty["text"])
        self.assertTrue(warranty["section"].startswith("2."))

    def test_long_section_repeats_title(self):
        text = "## Длинный вопрос про запас мощности блока?\n\n" + ("слово " * 800)
        chunks = chunk_knowledge_text(text, chunk_size=500, overlap=40)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertIn("Длинный вопрос про запас мощности блока", chunk["text"])
            self.assertIn("Длинный вопрос", chunk["section"])


class SearchTests(unittest.TestCase):
    def test_distance_threshold_drops_weak_hits(self):
        hits = format_search_hits({
            "ids": [["near", "far"]],
            "documents": [["близкий фрагмент", "чужая тема"]],
            "distances": [[0.2, 0.9]],
            "metadatas": [[
                {"source": "base.docx", "section": "Питание"},
                {"source": "base.docx", "section": "Другое"},
            ]],
        })
        kept = filter_by_distance(hits, 0.65)
        self.assertEqual([item["id"] for item in kept], ["near"])
        self.assertEqual(kept[0]["section"], "Питание")
        self.assertEqual(kept[0]["source"], "base.docx")

    def test_threshold_is_inclusive_and_keeps_missing_distance(self):
        documents = [
            {"id": "edge", "distance": 0.65, "text": "a"},
            {"id": "unknown", "distance": None, "text": "b"},
            {"id": "worse", "distance": 0.66, "text": "c"},
        ]
        kept = filter_by_distance(documents, 0.65)
        self.assertEqual([item["id"] for item in kept], ["edge", "unknown"])

    def test_signature_changes_with_chunker(self):
        signature = collection_signature("abc", "local:model")
        self.assertIn(f"chunker:{CHUNKER_VERSION}", signature)
        self.assertIn("abc", signature)


class CacheTests(unittest.TestCase):
    def test_exact_match_normalizes_case_and_spaces(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = RAGCache(str(Path(folder) / "cache.db"), ttl_hours=0, min_similarity=0.93)
            cache.set("Какой блок питания?", "Блок на 100 Вт", [{"text": "72 Вт", "section": "1"}])
            found = cache.get("  какой   блок питания? ")
            self.assertEqual(found["answer"], "Блок на 100 Вт")
            self.assertEqual(found["context"][0]["section"], "1")
            self.assertEqual(found["context"][0]["text"], "72 Вт")

    def test_legacy_string_context_becomes_document(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = RAGCache(str(Path(folder) / "cache.db"), ttl_hours=0)
            cache.set("Вопрос", "Ответ", ["старый текст"])
            found = cache.get("Вопрос")
            self.assertEqual(found["context"][0]["text"], "старый текст")
            self.assertEqual(found["context"][0]["section"], "")

    def test_expired_entry_is_removed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cache.db"
            cache = RAGCache(str(path), ttl_hours=1)
            cache.set("Вопрос", "Старый ответ")
            old = (datetime.now() - timedelta(hours=5)).isoformat()
            _sql(path, "UPDATE cache SET created_at = ?", (old,))
            self.assertIsNone(cache.get("Вопрос"))
            self.assertEqual(cache.get_stats()["total_entries"], 0)

    def test_zero_ttl_keeps_old_entry(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cache.db"
            cache = RAGCache(str(path), ttl_hours=0)
            cache.set("Вопрос", "Ответ")
            old = (datetime.now() - timedelta(days=30)).isoformat()
            _sql(path, "UPDATE cache SET created_at = ?", (old,))
            self.assertEqual(cache.get("Вопрос")["answer"], "Ответ")

    def test_similar_question_uses_embedding(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = RAGCache(str(Path(folder) / "cache.db"), ttl_hours=0, min_similarity=0.93)
            cache.set(
                "Можно ли резать ленту где угодно?",
                "Только по меткам",
                [{"text": "метки", "section": "10"}],
                embedding=[1.0, 0.0],
            )
            found = cache.get_similar([0.99, 0.01])
            self.assertEqual(found["answer"], "Только по меткам")
            self.assertEqual(found["context"][0]["section"], "10")
            self.assertIsNone(cache.get_similar([0.0, 1.0]))

    def test_old_database_gets_embedding_column(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cache.db"
            _sql(path, """
                CREATE TABLE cache (
                    query_hash TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    context TEXT,
                    created_at TIMESTAMP
                )
            """)
            cache = RAGCache(str(path), ttl_hours=0)
            cache.set("Вопрос", "Ответ", embedding=[1.0, 0.0])
            self.assertEqual(cache.get("Вопрос")["answer"], "Ответ")
            self.assertIsNotNone(cache.get_similar([1.0, 0.0]))

    def test_clear(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = RAGCache(str(Path(folder) / "cache.db"), ttl_hours=0)
            cache.set("Вопрос", "Ответ")
            cache.clear()
            self.assertIsNone(cache.get("Вопрос"))


class DialogueTests(unittest.TestCase):
    def test_followup_includes_previous_question(self):
        history = [{"query": "Какая степень защиты нужна в ванной?", "answer": "IP65"}]
        self.assertTrue(is_followup("а для улицы?"))
        self.assertFalse(is_followup("Какая гарантия на блоки питания?"))
        self.assertEqual(
            retrieval_query("а для улицы?", history),
            "Какая степень защиты нужна в ванной?. а для улицы?",
        )
        self.assertEqual(
            retrieval_query("Какая гарантия на блоки питания?", history),
            "Какая гарантия на блоки питания?",
        )

    def test_prompt_contains_history_and_section(self):
        prompt = build_prompt(
            "а для улицы?",
            [{"text": "Для улицы подходит IP67.", "section": "6. Степень защиты"}],
            [{"query": "Какая IP для ванной?", "answer": "IP65"}],
        )
        self.assertIn("Какая IP для ванной?", prompt)
        self.assertIn("6. Степень защиты", prompt)
        self.assertIn("IP67", prompt)
        self.assertIn("а для улицы?", prompt)

    def test_paths_do_not_depend_on_working_directory(self):
        resolved = Path(project_path("data"))
        self.assertTrue(resolved.is_absolute())
        self.assertEqual(resolved.name, "data")
        self.assertEqual(resolved.parent.name, "assistant_giga")

    def test_no_info_message_is_stable(self):
        self.assertIn("базе знаний", NO_INFO_ANSWER.lower())


class TelegramTests(unittest.TestCase):
    def test_chat_gets_only_the_answer(self):
        text = answer_for_telegram({
            "answer": "Нужен блок на 100 Вт.",
            "context_docs": [{
                "text": "фрагмент базы",
                "section": "1. Питание",
                "source": "base.docx",
            }],
            "model": "GigaChat",
        })
        self.assertEqual(text, "Нужен блок на 100 Вт.")
        self.assertNotIn("base.docx", text)
        self.assertNotIn("Питание", text)

    def test_long_answer_is_split_on_a_word(self):
        parts = split_telegram_message(("слово " * 40).strip(), limit=50)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(part) <= 50 for part in parts))
        self.assertTrue(all(part.startswith("слово") for part in parts))

    def test_each_chat_keeps_its_own_recent_turns(self):
        histories = {}
        remember_turn(histories, 1, "вопрос", "ответ")
        remember_turn(histories, 2, "другой", "второй")
        self.assertEqual(histories[2][0]["query"], "другой")
        for index in range(8):
            remember_turn(histories, 1, f"q{index}", f"a{index}")
        self.assertEqual(len(histories[1]), 6)
        self.assertEqual(histories[1][0]["query"], "q2")
        self.assertEqual(len(histories[2]), 1)

    def test_failed_delivery_keeps_the_same_update(self):
        self.assertEqual(advance_offset(4, 7, False), 4)
        self.assertEqual(advance_offset(4, 7, True), 8)


if __name__ == "__main__":
    unittest.main()
