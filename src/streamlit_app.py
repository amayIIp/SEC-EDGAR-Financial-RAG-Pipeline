# src/streamlit_app.py
# Root-level entrypoint that runs the main Streamlit UI from the demo package.
# This ensures backwards compatibility with any startup command pointed here.

import sys # Standard library module to modify path args.
from streamlit.web import cli as stcli # Streamlit CLI loader.

if __name__ == "__main__":
    # Redirect execution to the main streamlit app inside the demo directory.
    sys.argv = ["streamlit", "run", "src/demo/streamlit_app.py"]
    sys.exit(stcli.main())
