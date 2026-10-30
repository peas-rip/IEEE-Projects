"""Package for the web application.

Deliberately thin: `main` owns the FastAPI app, `routes` owns the endpoints,
`schemas` owns the wire contract, `predictions` owns the bundled artifact.
The model layer lives in `pneumonia.service` and never sees a Pydantic type.
"""
