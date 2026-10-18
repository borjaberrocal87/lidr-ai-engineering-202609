"""Pipeline de embeddings (Sesión 07).

Convierte presupuestos históricos en vectores: un chunker estructural (un
componente del presupuesto = un chunk) y un embedder de OpenAI. Los vectores se
generan en memoria y se devuelven por HTTP; la persistencia en una base
vectorial es territorio de la Sesión 08.
"""
