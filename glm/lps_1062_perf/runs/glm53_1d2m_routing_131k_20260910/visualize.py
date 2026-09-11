"""Simple distributions and rank totals from saved original-forward counts."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CONFIGS = {"CP8EP1": "#2962D9", "CP8EP8": "#E87924"}


def expert_bars(counts, rank, layer, config="CP8EP1"):
    frame = counts[(counts.config == config) & (counts.stage == "expert_input")
                   & (counts.moe_layer == layer)]
    selected = frame[frame["rank"] == rank].sort_values("expert")
    assert len(selected) == (256 if config == "CP8EP1" else 32)
    maximum = frame[frame["rank"].isin([0, 1])].tokens.max()
    fig = go.Figure(go.Bar(x=selected.expert.tolist(), y=selected.tokens.tolist(), marker_color=CONFIGS[config],
                          hovertemplate="Expert %{x}<br>%{y:,.0f} tokens<extra></extra>"))
    nonzero = selected[selected.tokens > 0]
    fig.add_trace(go.Scatter(
        x=nonzero.expert.tolist(),
        y=nonzero.tokens.tolist(),
        mode="markers",
        marker=dict(symbol="star", size=8, color="#1A1A1A", line=dict(color="white", width=0.75)),
        hovertemplate="Expert %{x}<br>%{y:,.0f} tokens<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(title=f"{config} · Rank {rank} · MoE {layer}", template="plotly_white",
                      height=340, autosize=True, bargap=0.08, showlegend=False,
                      margin=dict(l=65, r=20, t=55, b=55),
                      xaxis=dict(title="Expert ID", range=[int(selected.expert.min()) - 0.5, int(selected.expert.max()) + 0.5], dtick=32 if len(selected) == 256 else 4),
                      yaxis=dict(title="Routed tokens", range=[0, float(maximum) * 1.08], tickformat=","))
    return fig


def _canvas(title):
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.23,
                        subplot_titles=["MoE 1 — first MoE block", "MoE 2 — second MoE block"])
    fig.update_layout(title=dict(text=title, x=0.02, y=0.98, yanchor="top"), template="plotly_white", height=710, autosize=True,
                      barmode="group", font=dict(size=13),
                      legend=dict(orientation="h", y=1.13, x=0),
                      margin=dict(l=70, r=20, t=120, b=60))
    return fig


def expert_histograms(counts):
    fig = _canvas("Tokens per expert")
    edges = [0, 1, 32, 128, 512, 2048, 8192, np.inf]
    labels = ["0", "1–31", "32–127", "128–511", "512–2,047", "2,048–8,191", "8,192+"]
    largest = 0
    for row, layer in enumerate((1, 2), 1):
        for config, color in CONFIGS.items():
            values = counts[(counts.stage == "expert_input") & (counts.config == config)
                            & (counts.moe_layer == layer)].tokens.to_numpy()
            bins, _ = np.histogram(values, bins=edges)
            percentages = 100 * bins / len(values)
            largest = max(largest, percentages.max())
            fig.add_trace(go.Bar(name=config, legendgroup=config, showlegend=row == 1,
                                 x=labels, y=percentages, marker_color=color,
                                 text=[f"{v:.1f}%" for v in percentages], textposition="outside", textfont_size=11,
                                 cliponaxis=False, customdata=bins,
                                 hovertemplate="%{x} tokens<br>%{y:.1f}% of expert workloads<br>%{customdata} expert/GPU pairs<extra>%{fullData.name}</extra>"), row=row, col=1)
    fig.update_yaxes(title_text="Expert workloads (%)", range=[0, largest * 1.24], ticksuffix="%")
    fig.update_xaxes(title_text="Tokens handled by ONE expert on ONE GPU", categoryorder="array", categoryarray=labels)
    return fig


def gpu_load_bars(counts):
    fig = _canvas("Routed tokens per GPU")
    largest = 0
    for row, layer in enumerate((1, 2), 1):
        for config, color in CONFIGS.items():
            totals = (counts[(counts.stage == "expert_input") & (counts.config == config)
                             & (counts.moe_layer == layer)].groupby("rank").tokens.sum().reindex(range(8)))
            largest = max(largest, totals.max())
            fig.add_trace(go.Bar(name=config, legendgroup=config, showlegend=row == 1,
                                 x=[f"GPU {r}" for r in range(8)], y=totals, marker_color=color,
                                 text=[f"{v / 1000:.0f}k" for v in totals], textposition="outside", textfont_size=11,
                                 cliponaxis=False, hovertemplate="%{x}<br>%{y:,.0f} routed tokens<extra>%{fullData.name}</extra>"), row=row, col=1)
    fig.update_yaxes(title_text="Routed tokens", range=[0, largest * 1.18], tickformat="~s")
    return fig
