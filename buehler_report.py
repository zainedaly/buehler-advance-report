"""
Advance scouting report builder
Statcast (via pybaseball) -> one-page, self-contained HTML.

Run:  python buehler_report.py
Out:  index.html (images embedded, no other files needed)
"""
import base64
import io
from datetime import date
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from pybaseball import cache, statcast_pitcher

cache.enable()

# ----------------------------------------------------------------------
# CONFIG: edit these
# ----------------------------------------------------------------------
PITCHER_ID = 621111            # MLBAM ID. Verify on Baseball Savant before trusting.
PITCHER_NAME = "Walker Buehler"
THROWS = "RHP"
TEAM = "San Diego Padres"
GAME_CONTEXT = "NLDS Game 4 vs. Milwaukee Brewers"
START, END = "2026-03-01", "2026-10-06"
GAME_TYPES = ["R"]             # add "D" to include postseason starts
MIN_N = 20                     # hide split cells with fewer pitches than this
OUT_FILE = "index.html"
LOCATION_PITCHES = ["Cutter", "Slider", "Changeup"]

ANALYSIS = [
    "Against lefties, Buehler leans on the cutter (28% of pitches), shies away from the sinker (15%), and barely uses the slider (7%) or sweeper (3%). Against righties it flips: the sinker (26%) and the slider/sweeper pair (28% combined) lead the way, and the changeup is almost nonexistent (1%).",
    "His three fastballs are the contact pitches and his secondary pitches are the swing-and-miss pitches. The cutter, four-seam, and sinker combine for roughly a 15% whiff rate, while the slider, sweeper, knuckle curve, and changeup combine for roughly 26%.",
    "Against righties, Buehler starts at-bats with a fastball variant 81% of the time, well above the 65% he throws across all counts, and goes back to one 84% of the time when he falls behind.",
    "Against lefties, the changeup is a real part of his plan (16% of pitches), and he works it down and away. Roughly four in ten finish below the zone, 31.5% sit off the outer edge compared to 8.2% off the inner edge, and swings at ones below the zone miss 38.7% of the time.",
]

PALETTE = {
    "4-Seam Fastball": "#D55E00",
    "Sinker": "#0072B2",
    "Cutter": "#999999",
    "Slider": "#E69F00",
    "Sweeper": "#56B4E9",
    "Curveball": "#000000",
    "Knuckle Curve": "#CC79A7",
    "Changeup": "#009E73",
    "Split-Finger": "#7f7f7f",
}
MARKERS = {
    "4-Seam Fastball": "o",
    "Sinker": "s",
    "Cutter": "^",
    "Slider": "D",
    "Sweeper": "v",
    "Curveball": "*",
    "Knuckle Curve": "P",
    "Changeup": "p",
    "Split-Finger": "h",
}

BUCKETS = ["First pitch", "Pitcher ahead", "Even", "Hitter ahead", "Two strikes"]

SWINGS = {"swinging_strike", "swinging_strike_blocked", "foul_tip", "foul",
          "hit_into_play", "foul_bunt", "missed_bunt"}
WHIFFS = {"swinging_strike", "swinging_strike_blocked", "foul_tip", "missed_bunt"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#888888",
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
})


def color(pitch):
    return PALETTE.get(pitch, "#999999")


# ----------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------
def load():
    df = statcast_pitcher(START, END, PITCHER_ID)
    df = df[df["game_type"].isin(GAME_TYPES)].copy()
    return df


def prep(df):
    df = df.dropna(subset=["pitch_name"]).copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["swing"] = df["description"].isin(SWINGS)
    df["whiff"] = df["description"].isin(WHIFFS)
    df["csw"] = df["description"].isin(WHIFFS | {"called_strike"})
    df["bbe"] = df["launch_speed"].notna()
    df["hard"] = df["launch_speed"] >= 95

    b, s = df["balls"], df["strikes"]
    conds = [(b == 0) & (s == 0), s == 2, s > b, b == s, b > s]
    choices = [BUCKETS[0], BUCKETS[4], BUCKETS[1], BUCKETS[2], BUCKETS[3]]
    df["bucket"] = np.select(conds, choices, default="Even")
    return df


def arsenal_table(df):
    g = df.groupby("pitch_name")
    t = pd.DataFrame({
        "N": g.size(),
        "Usage %": g.size() / len(df) * 100,
        "Velo": g["release_speed"].mean(),
        "Spin": g["release_spin_rate"].mean(),
        "HB (in)": g["pfx_x"].mean() * 12,
        "iVB (in)": g["pfx_z"].mean() * 12,
        "Whiff %": g["whiff"].sum() / g["swing"].sum() * 100,
        "CSW %": g["csw"].mean() * 100,
        "Hard-hit %": g["hard"].sum() / g["bbe"].sum() * 100,
    }).sort_values("N", ascending=False)
    t.index.name = "Pitch"
    return t


def table_html(t):
    out = t.copy()
    out["N"] = out["N"].astype(int)
    fmt = {
        "Usage %": "{:.1f}", "Velo": "{:.1f}", "Spin": "{:.0f}",
        "HB (in)": "{:.1f}", "iVB (in)": "{:.1f}",
        "Whiff %": "{:.1f}", "CSW %": "{:.1f}", "Hard-hit %": "{:.1f}",
    }
    for col, f in fmt.items():
        out[col] = out[col].map(lambda v, f=f: f.format(v) if pd.notna(v) else "n/a")
    return out.reset_index().to_html(index=False, border=0, classes="arsenal")


# ----------------------------------------------------------------------
# FIGURES
# ----------------------------------------------------------------------
def to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def fig_usage_hand(df, pitches):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    y = np.arange(len(pitches))
    h = 0.38
    for k, (hand, col) in enumerate([("L", "#4c78a8"), ("R", "#f58518")]):
        d = df[df["stand"] == hand]
        use = d["pitch_name"].value_counts(normalize=True).reindex(pitches).fillna(0) * 100
        pos = y + (k - 0.5) * h
        ax.barh(pos, use.values, h, color=col, label=f"vs {hand}HH (n={len(d)})")
        for yi, v in zip(pos, use.values):
            ax.text(v + 0.6, yi, f"{v:.0f}%", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(pitches)
    ax.invert_yaxis()
    ax.set_xlabel("Usage (%)")
    ax.set_title("Pitch usage by batter handedness")
    ax.legend(frameon=False, loc="lower right")
    return to_b64(fig)


def fig_trend(df):
    fb = df[df["pitch_type"].isin(["FF", "SI"])]
    if fb.empty:
        return ""
    main = fb["pitch_name"].value_counts().idxmax()
    g = fb[fb["pitch_name"] == main].groupby("game_date")["release_speed"].mean()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(g.index, g.values, marker="o", ms=4, lw=1.5, color=color(main))
    ax.axhline(g.mean(), color="#888888", lw=0.8, ls="--")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_ylabel("Avg velo (mph)")
    ax.set_title(f"{main} velocity by start")
    fig.autofmt_xdate(rotation=30)
    return to_b64(fig)


def count_heat(ax, d, pitches, title):
    ct = pd.crosstab(d["pitch_name"], d["bucket"]).reindex(index=pitches, columns=BUCKETS).fillna(0)
    tot = ct.sum()
    pct = ct / tot.replace(0, np.nan) * 100
    pct.loc[:, tot < MIN_N] = np.nan
    ax.imshow(pct.values, cmap="Blues", vmin=0, vmax=70, aspect="auto")
    for i in range(pct.shape[0]):
        for j in range(pct.shape[1]):
            v = pct.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 40 else "#222222")
    ax.set_xticks(range(len(BUCKETS)))
    ax.set_xticklabels([f"{b}\n(n={int(tot[b])})" for b in BUCKETS], fontsize=7.5)
    ax.set_yticks(range(len(pitches)))
    ax.set_yticklabels(pitches)
    ax.set_title(title)
    for sp in ax.spines.values():
        sp.set_visible(False)


def fig_count(df, pitches):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    count_heat(axes[0], df[df["stand"] == "L"], pitches, "Usage % by count vs LHH")
    count_heat(axes[1], df[df["stand"] == "R"], pitches, "Usage % by count vs RHH")
    fig.text(0.5, -0.04, f"Cells with fewer than {MIN_N} pitches in the count group are left blank.",
             ha="center", fontsize=8, color="#666666")
    fig.tight_layout()
    return to_b64(fig)


def fig_location(df, pitches):
    top = LOCATION_PITCHES
    sz_top, sz_bot = df["sz_top"].median(), df["sz_bot"].median()
    fig, axes = plt.subplots(2, len(top), figsize=(11, 7.6))
    axes = np.atleast_2d(axes)
    for r, hand in enumerate(["L", "R"]):
        for c, p in enumerate(top):
            ax = axes[r, c]
            d = df[(df["pitch_name"] == p) & (df["stand"] == hand)]
            miss = d[d["whiff"]]
            ax.scatter(d["plate_x"], d["plate_z"], s=14, alpha=0.35,
                       color=color(p), edgecolor="none")
            ax.scatter(miss["plate_x"], miss["plate_z"], s=26, marker="x",
                       color="black", linewidth=1)
            ax.add_patch(Rectangle((-0.708, sz_bot), 1.416, sz_top - sz_bot,
                                   fill=False, ec="#333333", lw=1.2))
            ax.set_xlim(-2, 2)
            ax.set_ylim(0, 4.5)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{p} vs {hand}HH (n={len(d)})", fontsize=9)
    fig.text(0.5, 0.01, "Catcher's view. Black x = swinging miss. Box = median strike zone.",
             ha="center", fontsize=8, color="#666666")
    fig.tight_layout()
    return to_b64(fig)


def fig_movement_release(df, pitches):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    # Left panel: movement, with a label on each cluster
    ax = axes[0]
    for p in pitches:
        d = df[df["pitch_name"] == p]
        ax.scatter(d["pfx_x"] * 12, d["pfx_z"] * 12, s=34, alpha=0.6,
                   color=color(p), marker=MARKERS.get(p, "o"),
                   edgecolor="white", linewidth=0.4, label=p)
        ax.text(d["pfx_x"].median() * 12, d["pfx_z"].median() * 12, p,
                fontsize=8.5, fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#333333", lw=0.6, alpha=0.9))
    ax.axhline(0, color="#aaaaaa", lw=0.8)
    ax.axvline(0, color="#aaaaaa", lw=0.8)
    ax.set_xlabel("Horizontal break (in, catcher's view)")
    ax.set_ylabel("Induced vertical break (in)")
    ax.set_title("Pitch movement")

    # Right panel: release point, same markers and colors
    ax = axes[1]
    for p in pitches:
        d = df[df["pitch_name"] == p]
        ax.scatter(d["release_pos_x"], d["release_pos_z"], s=34, alpha=0.6,
                   color=color(p), marker=MARKERS.get(p, "o"),
                   edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Release side (ft)")
    ax.set_ylabel("Release height (ft)")
    ax.set_title("Release point")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, markerscale=1.6, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return to_b64(fig)
# ----------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------
TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__NAME__ | Advance Scouting Report</title>
<style>
  :root { --ink:#1b1f24; --muted:#6b7280; --line:#e5e7eb; --accent:#1f4e79; --bg:#f7f8fa; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; line-height:1.5; }
  .page { max-width:980px; margin:0 auto; padding:32px 24px 48px; }
  header { border-bottom:3px solid var(--accent); padding-bottom:14px; margin-bottom:22px; }
  h1 { margin:0; font-size:28px; letter-spacing:-0.02em; }
  .sub { color:var(--muted); margin-top:4px; font-size:14px; }
  .chips { margin-top:10px; display:flex; flex-wrap:wrap; gap:8px; }
  .chip { background:#fff; border:1px solid var(--line); border-radius:999px; padding:3px 12px; font-size:12px; color:var(--muted); }
  section { background:#fff; border:1px solid var(--line); border-radius:10px; padding:18px 20px; margin-bottom:18px; }
  h2 { margin:0 0 10px; font-size:13px; text-transform:uppercase; letter-spacing:0.08em; color:var(--accent); }
  img { max-width:100%; height:auto; display:block; margin:0 auto; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
  table.arsenal { width:100%; border-collapse:collapse; font-size:13px; }
  table.arsenal th { text-align:right; font-weight:600; color:var(--muted); border-bottom:1px solid var(--line); padding:6px 8px; }
  table.arsenal td { text-align:right; padding:6px 8px; border-bottom:1px solid #f0f1f3; }
  table.arsenal th:first-child, table.arsenal td:first-child { text-align:left; font-weight:600; }
  .analysis { border-left:4px solid var(--accent); }
  .analysis ul { margin:0; padding-left:18px; }
  .analysis li { margin-bottom:6px; }
  footer { color:var(--muted); font-size:11.5px; margin-top:8px; }
  @media (max-width:760px) { .grid { grid-template-columns:1fr; } }
  @media print { body { background:#fff; } section { break-inside:avoid; } }
</style>
</head>
<body>
<div class="page">
  <header>
    <h1>__NAME__</h1>
    <div class="sub">__THROWS__ | __TEAM__ | __CONTEXT__</div>
    <div class="chips">
      <span class="chip">__PITCHES__ pitches</span>
      <span class="chip">__GAMES__ games</span>
      <span class="chip">__RANGE__</span>
      <span class="chip">Source: Baseball Savant</span>
    </div>
  </header>

  <section class="analysis">
    <h2>Key takeaways</h2>
    <ul>__ANALYSIS__</ul>
  </section>

  <section><h2>Arsenal</h2>__TABLE__</section>

  <div class="grid">
    <section><h2>Usage by hand</h2><img src="data:image/png;base64,__USAGE__" alt="Usage by hand"></section>
    <section><h2>Recent form</h2><img src="data:image/png;base64,__TREND__" alt="Velocity trend"></section>
  </div>

  <section><h2>Usage by count</h2><img src="data:image/png;base64,__COUNT__" alt="Usage by count"></section>
  <section><h2>Location</h2><img src="data:image/png;base64,__LOC__" alt="Pitch locations"></section>
  <section><h2>Movement and release</h2><img src="data:image/png;base64,__MOVE__" alt="Movement and release"></section>

  <footer>
    Independent project built from public Statcast data. Not affiliated with or produced for any club.
    Whiff % = swinging strikes / swings. CSW % = called strikes plus whiffs / pitches.
    Hard-hit % = batted balls 95+ mph / tracked batted balls. Generated __TODAY__.
  </footer>
</div>
</body>
</html>
"""


def render(df, t, figs):
    html = TEMPLATE
    repl = {
        "__NAME__": PITCHER_NAME,
        "__THROWS__": THROWS,
        "__TEAM__": TEAM,
        "__CONTEXT__": GAME_CONTEXT,
        "__PITCHES__": f"{len(df):,}",
        "__GAMES__": str(df["game_pk"].nunique()),
        "__RANGE__": f"{df['game_date'].min():%b %d} to {df['game_date'].max():%b %d, %Y}",
        "__ANALYSIS__": "".join(f"<li>{a}</li>" for a in ANALYSIS),
        "__TABLE__": table_html(t),
        "__USAGE__": figs["usage"],
        "__TREND__": figs["trend"],
        "__COUNT__": figs["count"],
        "__LOC__": figs["loc"],
        "__MOVE__": figs["move"],
        "__TODAY__": date.today().strftime("%b %d, %Y"),
    }
    for k, v in repl.items():
        html = html.replace(k, v)
    return html


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    df = prep(load())

    # Team check
    df["pitching_team"] = np.where(df["inning_topbot"] == "Top", df["home_team"], df["away_team"])
    print(df["pitching_team"].value_counts())
    print(df.groupby("game_pk").size().describe())

    # Sanity check: read this before trusting anything below.
    print("Pitches:", len(df), "| Games:", df["game_pk"].nunique())
    print("Date range:", df["game_date"].min().date(), "to", df["game_date"].max().date())
    print(df["pitch_name"].value_counts())

    t = arsenal_table(df)
    pitches = t.index.tolist()

    # Sample sizes and release point spread per pitch
    print(df.groupby("pitch_name").agg(
        rel_z_mean=("release_pos_z", "mean"),
        rel_z_sd=("release_pos_z", "std"),
        bbe=("bbe", "sum"),
        swings=("swing", "sum"),
    ).round(2))

    ch = df[(df["stand"] == "L") & (df["pitch_name"] == "Changeup")].dropna(subset=["plate_x", "plate_z"])
    zb = df["sz_bot"].median()
    low = ch[ch["plate_z"] < zb]
    print("LHH changeups:", len(ch))
    print("Outside of plate (%):", round((ch["plate_x"] < -0.71).mean() * 100, 1))
    print("Inside of plate (%):", round((ch["plate_x"] > 0.71).mean() * 100, 1))
    print("Below zone (%):", round((ch["plate_z"] < zb).mean() * 100, 1))
    print("Whiff % below zone:", round(low["whiff"].sum() / low["swing"].sum() * 100, 1))

    figs = {
        "usage": fig_usage_hand(df, pitches),
        "trend": fig_trend(df),
        "count": fig_count(df, pitches),
        "loc": fig_location(df, pitches),
        "move": fig_movement_release(df, pitches),
    }
    Path(OUT_FILE).write_text(render(df, t, figs), encoding="utf-8")
    print(f"Wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
