"""Cuándo comprimir, qué comprimir y qué conservar literal.

La política es la única pieza que muta el ``ConversationHistory`` más allá de
``append``. Corre tras cada turno (ver ``EstimationService.estimate_conversational``)
con este bucle, en orden:

1. Mientras la ventana deslizante supere el tope (``max_turns * 2`` mensajes),
   despega del frente el par usuario/asistente más antiguo.
2. Envía el lado del usuario del par al ``AnchorDetector``. Si es ancla, mueve
   AMBOS mensajes del par a ``history.anchors`` para que el compromiso sobreviva
   literal. Si no, encola ambos para el resumidor.
3. Tras el bucle, si se despegaron pares no-ancla, se los entrega al
   ``CumulativeSummarizer`` junto con el resumen previo, y reemplaza
   ``history.summary`` con el nuevo texto rodante.

La política es intencionadamente idempotente: una segunda llamada sin cambios en
``messages`` es un no-op.
"""

from __future__ import annotations

import structlog

from app.services.llm_wrapper import LLMWrapper
from app.sessions.compression.anchors import AnchorDetector
from app.sessions.compression.summarizer import CumulativeSummarizer
from app.sessions.models import ConversationHistory, Message

log = structlog.get_logger()


class CompressionPolicy:
    def __init__(
        self,
        *,
        anchor_detector: AnchorDetector,
        summarizer: CumulativeSummarizer,
    ) -> None:
        self.anchor_detector = anchor_detector
        self.summarizer = summarizer

    def should_compress(self, history: ConversationHistory) -> bool:
        return len(history.messages) > history.max_turns * 2

    def apply(self, history: ConversationHistory) -> None:
        """Muta ``history`` in place: promueve anclas y absorbe el resto en el
        resumen rodante. No-op cuando la ventana está por debajo del tope."""

        if not self.should_compress(history):
            return

        evicted_for_summary: list[Message] = []
        promoted_anchor_rules: list[list[str]] = []

        while len(history.messages) > history.max_turns * 2:
            # Pop seguro por pares: con longitud par y roles alternos, las dos
            # entradas más antiguas son el par usuario/asistente a retirar.
            if len(history.messages) < 2:
                break
            user_msg = history.messages[0]
            assistant_msg = history.messages[1]

            match = self.anchor_detector.detect(user_msg)
            if match.is_anchor:
                history.anchors.append(user_msg)
                history.anchors.append(assistant_msg)
                promoted_anchor_rules.append(match.matched_rules)
            else:
                evicted_for_summary.append(user_msg)
                evicted_for_summary.append(assistant_msg)

            del history.messages[:2]

        if evicted_for_summary:
            history.summary = self.summarizer.summarize(
                previous_summary=history.summary,
                evicted=evicted_for_summary,
            )

        log.info(
            "history_compressed",
            promoted_anchors=len(promoted_anchor_rules),
            anchor_rules=[r for rules in promoted_anchor_rules for r in rules],
            evicted_to_summary=len(evicted_for_summary),
            summary_chars=len(history.summary or ""),
            anchors_count=len(history.anchors),
            recent_messages=len(history.messages),
        )


def apply_compression(
    history: ConversationHistory,
    *,
    llm_wrapper: LLMWrapper,
    compression_model: str,
    anchor_detection_mode: str = "heuristic",
) -> None:
    """Wrapper de conveniencia usado por ``EstimationService``.

    Construye el detector + resumidor con el cableado por defecto y ejecuta la
    política una vez. Se mantiene estrecho para que el servicio no tenga que
    conocer la estructura interna del módulo de compresión.
    """
    detector = AnchorDetector(
        mode=anchor_detection_mode,  # type: ignore[arg-type]
        llm_wrapper=llm_wrapper,
        llm_model=compression_model,
    )
    summarizer = CumulativeSummarizer(llm_wrapper=llm_wrapper, model=compression_model)
    policy = CompressionPolicy(anchor_detector=detector, summarizer=summarizer)
    policy.apply(history)
