"""Execute the visualization notebook and save static fallbacks for Cursor."""
import base64
from pathlib import Path

import nbformat
import pandas as pd
from nbclient import NotebookClient
from visualize import expert_bars

root = Path(__file__).parent
path = root / "routing_counts.ipynb"
notebook = nbformat.read(path, as_version=4)
NotebookClient(notebook, timeout=180, kernel_name="glm-routing-visualization", resources={"metadata": {"path": str(root)}}).execute()
counts = pd.read_csv(root / "counts.csv")
cases = [(config, rank, layer) for config in ("CP8EP1", "CP8EP8") for rank in (0, 1) for layer in (1, 2)]
for cell_index, (config, rank, layer) in enumerate(cases, 2):
    name = f"{config.lower()}.rank{rank}.moe{layer}"
    figure = expert_bars(counts, rank, layer, config)
    png = figure.to_image(format="png", width=950, height=340, scale=1.5)
    (root / f"{name}.png").write_bytes(png)
    figure.write_html(root / f"{name}.html", include_plotlyjs=True, config={"responsive": True})
    for output in notebook.cells[cell_index].outputs:
        if output.output_type in ("display_data", "execute_result"):
            output.data["image/png"] = base64.b64encode(png).decode()
nbformat.write(notebook, path)
print(path)
