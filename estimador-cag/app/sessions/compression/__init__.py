"""Compresión híbrida de memoria para conversaciones largas.

Tres piezas trabajan juntas:

- ``AnchorDetector`` decide si un turno porta información durable que la
  conversación no debería olvidar nunca (NDA firmado, alcance congelado,
  presupuesto acordado, mención regulatoria). Las anclas se excluyen del desalojo.
- ``CumulativeSummarizer`` funde turnos antiguos que no son ancla en un resumen
  de texto libre rodante, guardado en el ``ConversationHistory``.
- ``CompressionPolicy`` es el orquestador: inspecciona el historial tras cada
  ``append`` y decide qué (si algo) comprimir.

La salida de ``to_messages()`` tras la compresión es:

    [resumen_sintético?] + anclas_en_orden + ventana_reciente_deslizante
"""

from app.sessions.compression.anchors import AnchorDetector, AnchorMatch
from app.sessions.compression.policy import CompressionPolicy, apply_compression
from app.sessions.compression.summarizer import CumulativeSummarizer

__all__ = [
    "AnchorDetector",
    "AnchorMatch",
    "CompressionPolicy",
    "CumulativeSummarizer",
    "apply_compression",
]
