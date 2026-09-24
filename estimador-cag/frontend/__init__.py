"""Capa de presentación (frontend) del estimador CAG.

Este paquete habla con el backend FastAPI únicamente por HTTP: no importa
`app.*`. Cambiar Streamlit por otra UI implica reescribir el punto de entrada
y reutilizar `frontend.client`.
"""
