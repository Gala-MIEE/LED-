"""
Модуль работы с векторным хранилищем ChromaDB для GigaChat.
Загружает документы Word, режет их на фрагменты и ищет ближайшие.
"""

import hashlib
import logging
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

logger = logging.getLogger("assistant_giga.vector_store")

_WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
LOCAL_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CHUNKER_VERSION = "2"
# Косинусная дистанция Chroma: 0 — тот же смысл, чем больше — тем дальше.
# 0.65 отсекает слабые совпадения и оставляет перефразированные вопросы из базы.
DEFAULT_MAX_DISTANCE = 0.65


def project_path(path: str) -> str:
    """Путь относительно папки assistant_giga, если он не абсолютный."""
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str(Path(__file__).resolve().parent / candidate)


def collection_signature(file_hash: str, backend_id: str) -> str:
    """Подпись индекса: файл, модель эмбеддингов и версия нарезки."""
    return f"{file_hash}|{backend_id}|chunker:{CHUNKER_VERSION}"


def cosine_similarity(left: List[float], right: List[float]) -> float:
    """Косинусная близость двух векторов. 1 — одинаковое направление."""
    if not left or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / ((left_norm ** 0.5) * (right_norm ** 0.5))


def filter_by_distance(documents: List[Dict[str, Any]], max_distance: float) -> List[Dict[str, Any]]:
    """Оставляет фрагменты не дальше порога. Запись без дистанции не отбрасывается."""
    kept = []
    for document in documents:
        distance = document.get("distance")
        if distance is None or float(distance) <= max_distance:
            kept.append(document)
    return kept


def format_search_hits(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Приводит ответ ChromaDB к списку фрагментов с разделом и файлом."""
    found = results.get("documents") or []
    if not found or not found[0]:
        return []

    ids = (results.get("ids") or [[]])[0]
    raw_distances = results.get("distances")
    distances = raw_distances[0] if raw_distances else [None] * len(found[0])
    raw_metadatas = results.get("metadatas")
    metadatas = raw_metadatas[0] if raw_metadatas else [{}] * len(found[0])

    documents = []
    for index, text in enumerate(found[0]):
        meta = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        if not isinstance(meta, dict):
            meta = {}
        distance = distances[index] if index < len(distances) else None
        documents.append({
            "id": ids[index] if index < len(ids) else "",
            "text": text,
            "distance": None if distance is None else float(distance),
            "source": str(meta.get("source", "") or ""),
            "section": str(meta.get("section", "") or ""),
        })
    return documents


class LocalEmbedder:
    """Локальные эмбеддинги. Модель скачивается один раз и дальше работает без API."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("LOCAL_EMBEDDING_MODEL") or LOCAL_EMBEDDING_MODEL
        self._model = None

    def backend_id(self) -> str:
        return f"local:{self.model_name}"

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            print(f"Загрузка локальной модели эмбеддингов {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
            print("Локальная модель эмбеддингов готова")
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self._load().encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]


def resolve_data_file(data_file: str = "data") -> str:
    """
    Находит документ базы знаний.
    Приоритет у файла Word (.docx или .doc) в папке data.
    """
    folder = data_file if os.path.isdir(data_file) else (os.path.dirname(data_file) or ".")
    word_files = []
    if os.path.isdir(folder):
        for name in sorted(os.listdir(folder)):
            if name.startswith("~$"):
                continue
            if name.lower().endswith((".docx", ".doc")):
                word_files.append(os.path.join(folder, name))
    if word_files:
        return word_files[0]
    if os.path.isfile(data_file):
        return data_file
    raise FileNotFoundError(
        f"В папке {folder} нет файла .doc или .docx с базой знаний"
    )


def read_text_document(file_path: str) -> str:
    """Читает текст из .docx/.doc (Office Open XML) или обычного текстового файла."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in {".docx", ".doc"}:
        if not zipfile.is_zipfile(file_path):
            raise ValueError(
                f"Файл {file_path} в старом двоичном формате .doc. "
                "Сохраните его как .docx и положите в папку data."
            )
        return _read_docx(file_path)

    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def _style_name_map(styles_xml: bytes) -> Dict[str, str]:
    """Сопоставляет id стиля Word с его именем."""
    root = ET.fromstring(styles_xml)
    names = {}
    for style in root.iter(f"{_WORD_NS}style"):
        style_id = style.get(f"{_WORD_NS}styleId")
        name_node = style.find(f"{_WORD_NS}name")
        if style_id and name_node is not None:
            names[style_id] = name_node.get(f"{_WORD_NS}val") or ""
    return names


def _heading_level(style_id: str, style_name: str) -> Optional[int]:
    """Уровень заголовка Word: 1–6. Для обычного абзаца — None."""
    for raw in (style_id, style_name):
        if not raw:
            continue
        compact = raw.lower().replace(" ", "").replace("-", "")
        if compact in {"title", "heading", "заголовок"}:
            return 1
        for level in range(1, 7):
            if compact in {f"heading{level}", f"заголовок{level}"}:
                return level
    return None


def _paragraph_text(paragraph) -> str:
    parts = []
    for node in paragraph.iter():
        if node.tag == f"{_WORD_NS}t" and node.text:
            parts.append(node.text)
        elif node.tag == f"{_WORD_NS}tab":
            parts.append("\t")
        elif node.tag == f"{_WORD_NS}br":
            parts.append("\n")
    return "".join(parts).strip()


def _paragraph_style_id(paragraph) -> str:
    properties = paragraph.find(f"{_WORD_NS}pPr")
    if properties is None:
        return ""
    style = properties.find(f"{_WORD_NS}pStyle")
    if style is None:
        return ""
    return style.get(f"{_WORD_NS}val") or ""


def _format_paragraph(text: str, style_id: str, style_name: str) -> str:
    """Сохраняет заголовок Word как строку Markdown, не дублируя уже готовые #."""
    level = _heading_level(style_id, style_name)
    if level and not text.lstrip().startswith("#"):
        return f"{'#' * level} {text}"
    return text


def _table_text(table) -> str:
    """Читает строки таблицы как текст «ячейка | ячейка»."""
    rows = []
    for row in table.findall(f"{_WORD_NS}tr"):
        cells = []
        for cell in row.findall(f"{_WORD_NS}tc"):
            cell_lines = []
            for paragraph in cell.findall(f"{_WORD_NS}p"):
                text = _paragraph_text(paragraph)
                if text:
                    cell_lines.append(text)
            cells.append(" ".join(cell_lines))
        if any(cell.strip() for cell in cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _read_docx(file_path: str) -> str:
    """Достаёт абзацы и таблицы из Word в порядке документа."""
    with zipfile.ZipFile(file_path) as archive:
        xml = archive.read("word/document.xml")
        try:
            style_names = _style_name_map(archive.read("word/styles.xml"))
        except KeyError:
            style_names = {}

    root = ET.fromstring(xml)
    body = root.find(f"{_WORD_NS}body")
    if body is None:
        raise ValueError(f"В файле {file_path} нет текста")

    blocks = []
    for child in list(body):
        if child.tag == f"{_WORD_NS}p":
            text = _paragraph_text(child)
            if not text:
                continue
            style_id = _paragraph_style_id(child)
            blocks.append(_format_paragraph(text, style_id, style_names.get(style_id, "")))
        elif child.tag == f"{_WORD_NS}tbl":
            table = _table_text(child)
            if table:
                blocks.append(table)

    if not blocks:
        raise ValueError(f"В файле {file_path} нет текста")
    return "\n\n".join(blocks)


def _split_markdown_sections(text: str) -> List[str]:
    """Делит текст по строкам заголовков Markdown."""
    parts = re.split(r"\n+(?=#{1,6}\s+)", text)
    return [part.strip() for part in parts if part.strip()]


def _section_title(section: str) -> str:
    for line in section.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""


def _get_overlap_text(text: str, overlap_size: int) -> str:
    """Берёт хвост чанка, по возможности с границы предложения."""
    if len(text) <= overlap_size:
        return text

    overlap_candidate = text[-overlap_size:]
    sentence_starts = [". ", "! ", "? ", "\n"]
    best_start = 0

    for delimiter in sentence_starts:
        position = overlap_candidate.find(delimiter)
        if position != -1 and position > best_start:
            best_start = position + len(delimiter)

    if best_start > 0:
        return overlap_candidate[best_start:].strip()
    return overlap_candidate.strip()


def _hard_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Режет фрагмент без знаков конца предложения, чтобы он не оставался одним куском."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        next_start = end - overlap if 0 < overlap < chunk_size else end
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks


def _split_long_paragraph(paragraph: str, chunk_size: int, overlap: int) -> List[str]:
    """Режет длинный абзац по предложениям."""
    sentences = re.split(r"([.!?]+\s+)", paragraph)
    full_sentences = []
    for index in range(0, len(sentences) - 1, 2):
        if index + 1 < len(sentences):
            full_sentences.append(sentences[index] + sentences[index + 1])
        else:
            full_sentences.append(sentences[index])
    if len(sentences) % 2 == 1:
        full_sentences.append(sentences[-1])

    chunks = []
    current_chunk = ""
    for sentence in full_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            pieces = _hard_split(sentence, chunk_size, overlap)
            if pieces:
                chunks.extend(pieces[:-1])
                current_chunk = pieces[-1]
            continue
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk = f"{current_chunk} {sentence}".strip()
        elif current_chunk:
            chunks.append(current_chunk)
            overlap_text = _get_overlap_text(current_chunk, overlap)
            current_chunk = f"{overlap_text} {sentence}".strip() if overlap_text else sentence
        else:
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def _chunk_paragraphs(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Запасная нарезка, если в тексте нет коротких разделов с заголовками."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if len(paragraph) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            sentence_chunks = _split_long_paragraph(paragraph, chunk_size, overlap)
            if sentence_chunks:
                chunks.extend(sentence_chunks[:-1])
                current_chunk = sentence_chunks[-1]
        elif len(current_chunk) + len(paragraph) + 2 <= chunk_size:
            current_chunk = f"{current_chunk}\n\n{paragraph}".strip() if current_chunk else paragraph
        elif current_chunk:
            chunks.append(current_chunk)
            overlap_text = _get_overlap_text(current_chunk, overlap)
            current_chunk = f"{overlap_text}\n\n{paragraph}".strip() if overlap_text else paragraph
        else:
            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)
    return [chunk for chunk in chunks if len(chunk) >= 50]


def chunk_knowledge_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict[str, str]]:
    """
    Режет базу на фрагменты.
    Раздел с заголовком остаётся целиком, чтобы вопрос не отрывался от ответа.
    Длинный раздел режется дальше, но заголовок повторяется в каждом куске.
    """
    sections = _split_markdown_sections(text)
    if len(sections) <= 1 and not _section_title(text):
        return [{"text": chunk, "section": ""} for chunk in _chunk_paragraphs(text, chunk_size, overlap)]

    chunks = []
    for section in sections:
        title = _section_title(section)
        if len(section) <= 2500:
            pieces = [section]
        else:
            pieces = _chunk_paragraphs(section, chunk_size, overlap)

        for piece in pieces:
            piece = piece.strip()
            if len(piece) < 50:
                continue
            body_lines = [
                line.strip() for line in piece.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            if not body_lines:
                continue
            if title and title not in piece:
                piece = f"## {title}\n\n{piece}"
            chunks.append({"text": piece, "section": title})
    return chunks


class VectorStore:
    """Векторное хранилище на основе ChromaDB с локальными эмбеддингами."""

    def __init__(self, collection_name: str = "rag_collection", persist_directory: str = "chroma_db"):
        """
        Инициализация векторного хранилища.

        Args:
            collection_name: имя коллекции в ChromaDB
            persist_directory: директория для хранения данных
        """
        import chromadb

        self.collection_name = collection_name
        self.persist_directory = project_path(persist_directory)
        os.makedirs(self.persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_directory)

        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Коллекция '{collection_name}' загружена. Документов: {self.collection.count()}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            print(f"Создана новая коллекция '{collection_name}'")

        self.embedder = LocalEmbedder()

    def load_documents(self, file_path: str) -> bool:
        """
        Загружает документ в индекс.
        Возвращает True, если индекс собран заново и старый кеш ответов нужно сбросить.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл {file_path} не найден")

        signature = collection_signature(self._file_signature(file_path), self.embedder.backend_id())
        marker_path = self._source_marker_path()
        stored_signature = ""
        if os.path.exists(marker_path):
            with open(marker_path, "r", encoding="utf-8") as marker:
                stored_signature = marker.read().strip()

        if self.collection.count() > 0 and stored_signature == signature:
            print(f"Документы уже загружены из {file_path}")
            return False

        if self.collection.count() > 0 or stored_signature:
            print("Источник документов изменился, коллекция собирается заново")
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        text = read_text_document(file_path)
        chunks = chunk_knowledge_text(text)
        if not chunks:
            raise ValueError(f"В файле {file_path} не получилось выделить фрагменты")
        print(f"Текст из {os.path.basename(file_path)} разбит на {len(chunks)} фрагментов")

        print(f"Считаю эмбеддинги локальной моделью {self.embedder.model_name}...")
        documents = [chunk["text"] for chunk in chunks]
        embeddings = self.embedder.embed(documents)
        source_name = os.path.basename(file_path)
        metadatas = [
            {"source": source_name, "section": chunk["section"]}
            for chunk in chunks
        ]
        ids = [f"doc_{index}" for index in range(len(chunks))]

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )

        with open(marker_path, "w", encoding="utf-8") as marker:
            marker.write(signature)

        print(f"Загружено {len(chunks)} фрагментов из {source_name}")
        return True

    def embed_query(self, text: str) -> List[float]:
        """Вектор одного текста локальной моделью."""
        return self.embedder.embed([text])[0]

    def _file_signature(self, file_path: str) -> str:
        digest = hashlib.sha256()
        with open(file_path, "rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _source_marker_path(self) -> str:
        return os.path.join(self.persist_directory, f"{self.collection_name}.source")

    def search(self, query: str, top_k: int = 3, max_distance: float = None,
               query_embedding: List[float] = None) -> List[Dict[str, Any]]:
        """
        Ищет релевантные фрагменты и отбрасывает те, что дальше порога.

        Args:
            query: текст запроса
            top_k: сколько ближайших фрагментов смотреть
            max_distance: максимальная косинусная дистанция
            query_embedding: уже посчитанный вектор запроса
        """
        if max_distance is None:
            raw_distance = os.getenv("RAG_MAX_DISTANCE")
            max_distance = float(raw_distance) if raw_distance else DEFAULT_MAX_DISTANCE

        available = self.collection.count()
        if available == 0:
            return []

        if query_embedding is None:
            query_embedding = self.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, available),
        )
        hits = format_search_hits(results)
        kept = filter_by_distance(hits, max_distance)
        if hits and not kept:
            distances = [hit["distance"] for hit in hits if hit["distance"] is not None]
            best = min(distances) if distances else None
            logger.debug(
                "Нет фрагментов ближе порога %s, лучшая дистанция %s",
                max_distance,
                best,
            )
        return kept

    def get_collection_stats(self) -> Dict[str, Any]:
        """Статистика коллекции."""
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
            "persist_directory": self.persist_directory,
        }


if __name__ == "__main__":
    vector_store = VectorStore(collection_name="test_collection")

    data_file = resolve_data_file(project_path("data"))
    if os.path.exists(data_file):
        vector_store.load_documents(data_file)

    results = vector_store.search("Какая степень защиты IP нужна для ленты в ванной?", top_k=3)
    print("\nРезультаты поиска:")
    for index, doc in enumerate(results, 1):
        print(f"\n{index}. [{doc['section']}] {doc['text'][:200]}...")
        print(f"   Distance: {doc['distance']}")

    stats = vector_store.get_collection_stats()
    print(f"\nСтатистика: {stats}")
