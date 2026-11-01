"""
Оценка качества RAG-системы через RAGAS для assistant_giga.
И ответы, и метрики RAGAS считаются только через GigaChat.
"""

import asyncio
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

# RAGAS 0.4.3 импортирует ChatVertexAI из модуля, который убрали
# из новых версий langchain-community. Проект его не использует.
try:
    import langchain_community.chat_models.vertexai  # noqa: F401
except ModuleNotFoundError:
    from langchain_core.language_models.chat_models import BaseChatModel

    _vertexai_mod = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI(BaseChatModel):
        """Заглушка: Vertex AI в этом проекте не используется."""

        @property
        def _llm_type(self) -> str:
            return "chat-vertexai"

    _vertexai_mod.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_mod

from datasets import Dataset
from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import PromptValue
from ragas import evaluate
from ragas.llms.base import BaseRagasLLM

def _load_ragas_metrics():
    """Faithfulness, Context Precision и Context Recall, если он есть в установленной версии."""
    try:
        from ragas.metrics._faithfulness import Faithfulness
        from ragas.metrics._context_precision import ContextPrecision
        try:
            from ragas.metrics._context_recall import ContextRecall
            recall_metric = ContextRecall
        except ImportError:
            recall_metric = None
        return Faithfulness, ContextPrecision, recall_metric
    except ImportError:
        pass

    try:
        from ragas.metrics.collections import context_precision as precision_metric
        from ragas.metrics.collections import faithfulness as faithfulness_metric
        try:
            from ragas.metrics.collections import context_recall as recall_metric
        except ImportError:
            recall_metric = None
        return faithfulness_metric, precision_metric, recall_metric
    except ImportError:
        from ragas.metrics import context_precision as precision_metric
        from ragas.metrics import faithfulness as faithfulness_metric
        try:
            from ragas.metrics import context_recall as recall_metric
        except ImportError:
            recall_metric = None
        return faithfulness_metric, precision_metric, recall_metric


faithfulness, context_precision, context_recall = _load_ragas_metrics()

from gigachat_client import GigaChatClient, gigachat_credentials_set
from rag_pipeline import RAGPipeline


EVALUATION_SET = [
    (
        "Какой блок питания нужен для светодиодной ленты 12 В длиной 5 метров мощностью 14,4 Вт/м?",
        "Суммарная мощность 72 Вт, с запасом около 30% нужен блок питания 100 Вт на 12 В.",
    ),
    (
        "Можно ли подключать светодиодную ленту 24 В к блоку питания 12 В?",
        "Нет, нельзя. Напряжение блока питания должно совпадать с напряжением ленты.",
    ),
    (
        "Какая степень защиты IP нужна для ленты в ванной и на улице?",
        "Для ванной подходит IP65, для улицы — IP67 или IP68.",
    ),
    (
        "Можно ли резать светодиодную ленту в любом месте?",
        "Нет. Резать ленту можно только по специальным меткам.",
    ),
    (
        "Как оформить возврат или обмен светодиодной продукции?",
        "Возврат возможен в течение 14 дней, если товар не был в эксплуатации и сохранены упаковка и чек.",
    ),
    (
        "Сколько метров ленты 12 В можно подключить последовательно к одному блоку питания?",
        "Не более 5 метров. Для ленты 24 В — не более 10 метров, иначе дальний конец будет тусклым.",
    ),
    (
        "Какая гарантия действует на светодиодные ленты и блоки питания?",
        "На ленты и модули — 24 месяца, на блоки питания и драйверы — 12 месяцев, если иное не указано в карточке товара.",
    ),
    (
        "Какой индекс цветопередачи CRI нужен для витрины магазина?",
        "Для магазинов, витрин, салонов и музеев рекомендуется CRI не ниже 90. Для жилых помещений достаточно CRI не ниже 80.",
    ),
    (
        "Что означает цветовая температура 4000K?",
        "4000–4500K — это нейтральный белый свет, его используют для кухонь, офисов и магазинов.",
    ),
    (
        "Чем блок питания отличается от драйвера для светодиодов?",
        "Блок питания стабилизирует напряжение и нужен для лент. Драйвер стабилизирует ток и нужен для модулей и матриц без встроенной электроники.",
    ),
    (
        "Можно ли управлять светодиодной лентой со смартфона?",
        "Да, если между блоком питания и лентой поставить Wi-Fi или Bluetooth-контроллер, совместимый с напряжением ленты.",
    ),
    (
        "Как хранить светодиодную ленту?",
        "В заводской упаковке, в сухом помещении при температуре от −20 до +40 °C и влажности не более 70 %.",
    ),
    (
        "Как подобрать драйвер для модуля 700 мА и 32 В?",
        "Нужен драйвер на 700 мА мощностью около 25–30 Вт, диапазон напряжения которого включает 32 В.",
    ),
]


@dataclass
class GigaChatRagasLLM(BaseRagasLLM):
    """Языковая модель RAGAS, которая ходит только в GigaChat."""

    client: GigaChatClient = field(default=None, repr=False)
    model: str = "GigaChat"
    max_tokens: int = 1500

    def __post_init__(self):
        if self.client is None:
            self.client = GigaChatClient()
        super().__post_init__()

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.01,
        stop=None,
        callbacks=None,
    ) -> LLMResult:
        text = prompt.to_string()
        generations = []
        for _ in range(n):
            content = self.client.chat_completion(
                messages=[{"role": "user", "content": text}],
                model=self.model,
                temperature=0.01 if temperature is None else temperature,
                max_tokens=self.max_tokens,
            )
            generations.append(
                Generation(text=content, generation_info={"finish_reason": "stop"})
            )
        return LLMResult(generations=[generations])

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.01,
        stop=None,
        callbacks=None,
    ) -> LLMResult:
        return await asyncio.to_thread(
            self.generate_text,
            prompt,
            n,
            temperature,
            stop,
            callbacks,
        )

    def is_finished(self, response: LLMResult) -> bool:
        return True


def prepare_dataset(pipeline: RAGPipeline, samples: list) -> Dataset:
    """Собирает ответы GigaChat и контекст для RAGAS."""
    questions_list = []
    answers_list = []
    contexts_list = []
    ground_truths_list = []

    print("[*] Получение ответов от RAG-системы (GigaChat)...\n")

    for i, (question, ground_truth) in enumerate(samples, 1):
        print(f"  {i}/{len(samples)}: {question}")
        result = pipeline.query(question, use_cache=False)

        questions_list.append(question)
        answers_list.append(result["answer"])
        contexts_list.append([doc["text"] for doc in result["context_docs"]])
        ground_truths_list.append(ground_truth)
        print("     [+] Ответ получен от GigaChat")

    print()
    return Dataset.from_dict({
        "question": questions_list,
        "answer": answers_list,
        "contexts": contexts_list,
        "ground_truth": ground_truths_list,
    })


def evaluate_rag_system():
    """Оценивает RAG на GigaChat метриками Faithfulness и Context Precision."""
    print("=" * 70)
    print("ОЦЕНКА КАЧЕСТВА RAG-СИСТЕМЫ (GIGACHAT) ЧЕРЕЗ RAGAS")
    print("=" * 70)
    print()

    if not gigachat_credentials_set():
        print("[ОШИБКА] Ключ GigaChat не установлен")
        print("\nДобавьте в .env в корне проекта:")
        print("  GIGACHAT_AUTHORIZATION_KEY=your-auth-key")
        print("  GIGACHAT_SCOPE=GIGACHAT_API_PERS")
        print("  GIGACHAT_MODEL=GigaChat")
        sys.exit(1)

    try:
        print("[*] Инициализация RAG-системы (GigaChat)...\n")
        pipeline = RAGPipeline(
            collection_name="gigachat_rag_collection",
            cache_db_path="gigachat_rag_cache.db",
            data_file="data",
        )
        print("\n[OK] RAG-система готова к оценке\n")
    except Exception as e:
        print(f"[ОШИБКА] Ошибка инициализации RAG pipeline: {e}")
        sys.exit(1)

    print("=" * 70)
    dataset = prepare_dataset(pipeline, EVALUATION_SET)
    print("=" * 70)

    print("\n[*] Запуск оценки метрик RAGAS...")
    metric_names = "Faithfulness, Context Precision"
    if context_recall is not None:
        metric_names += ", Context Recall"
    print(f"   Метрики: {metric_names}")
    print("   Судья метрик: GigaChat")
    print("   (это займёт несколько минут)\n")

    metrics_to_use = [faithfulness(), context_precision()]
    if context_recall is not None:
        metrics_to_use.append(context_recall())
    judge = GigaChatRagasLLM(client=pipeline.gigachat_client, model=pipeline.model)

    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics_to_use,
            llm=judge,
        )
    except Exception as e:
        print(f"[ОШИБКА] Ошибка при оценке: {e}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 70)

    import math

    def _is_nan(value) -> bool:
        try:
            return math.isnan(float(value))
        except (TypeError, ValueError):
            return False

    metric_specs = [
        ("faithfulness", "Faithfulness (точность ответа)"),
        ("context_precision", "Context Precision (точность контекста)"),
    ]
    if context_recall is not None:
        metric_specs.append(("context_recall", "Context Recall (полнота контекста)"))

    def _column(name):
        try:
            return list(result[name])
        except Exception:
            return []

    averages = []
    print()
    print("[МЕТРИКИ] Средние значения:")
    for name, label in metric_specs:
        values = [value for value in _column(name) if not _is_nan(value)]
        average = sum(values) / len(values) if values else 0
        averages.append(average)
        print(f"   {label}: {average:.4f}")

    avg_score = sum(averages) / len(averages) if averages else 0
    print(f"\n{'─' * 70}")
    print(f"[ИТОГО] Средний балл: {avg_score:.4f}")

    if avg_score >= 0.7:
        print("   Оценка: Отличное качество! [OK]")
        print("   Система показывает высокую точность и релевантность ответов.")
    elif avg_score >= 0.5:
        print("   Оценка: Удовлетворительное качество [!]")
        print("   Рекомендуется улучшить качество документов или промптов.")
    else:
        print("   Оценка: Требует значительного улучшения [X]")
        print("   Необходимо пересмотреть стратегию chunking или качество данных.")

    print("\n" + "=" * 70)
    print("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ПО ВОПРОСАМ")
    print("=" * 70)

    for i, (question, _) in enumerate(EVALUATION_SET):
        print(f"\n{i + 1}. {question}")
        for name, label in metric_specs:
            column = _column(name)
            if i >= len(column) or _is_nan(column[i]):
                print(f"   {label}: не удалось вычислить")
            else:
                print(f"   {label}: {float(column[i]):.4f}")

    print("\n" + "=" * 70)
    print("[INFO] ПОЯСНЕНИЯ К МЕТРИКАМ")
    print("=" * 70)
    print("""
Faithfulness (Точность ответа):
  Измеряет, насколько ответ соответствует предоставленному контексту.
  Значения: 0.0 - 1.0 (1.0 = полное соответствие контексту)

Context Precision (Точность контекста):
  Измеряет качество извлечённого контекста для ответа на вопрос.
  Значения: 0.0 - 1.0 (1.0 = идеальный контекст)

Context Recall (Полнота контекста):
  Проверяет, что в найденных фрагментах есть сведения из эталонного ответа.
  Значения: 0.0 - 1.0 (1.0 = эталон полностью покрыт контекстом)
    """)

    print("=" * 70)
    print("[OK] Оценка завершена!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    evaluate_rag_system()
