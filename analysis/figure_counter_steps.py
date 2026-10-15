"""SVG figure for the blog post: the energy counter read back to back under load.

Draws the first 22 ms of <probe dir>/load-counter-changes.csv as a staircase
(counter value relative to the first read, in joules) and marks where a 10 ms
sampling thread would read it. Colors and fonts follow the portfolio site.

Usage: python figure_counter_steps.py <probe data dir> > counter-steps.svg
Standard library only.
"""

import csv
import sys
from pathlib import Path

RESOLUTION_J = 15.3e-6
WINDOW_MS = 22.0
X0, X1, Y0, Y1 = 70, 690, 300, 50  # plot area in SVG units
Y_MAX_J = 3.5


def x(ms):
    return X0 + (X1 - X0) * ms / WINDOW_MS


def y(joules):
    return Y0 - (Y0 - Y1) * joules / Y_MAX_J


def main():
    rows = list(csv.DictReader(open(Path(sys.argv[1]) / "load-counter-changes.csv", newline="")))
    t0, e0 = float(rows[0]["host_s"]), int(rows[0]["energy_count"])
    points = [((float(r["host_s"]) - t0) * 1e3, (int(r["energy_count"]) - e0) * RESOLUTION_J)
              for r in rows]
    points = [p for p in points if p[0] <= WINDOW_MS]

    stairs = []
    for (ta, ea), (tb, _) in zip(points, points[1:] + [(WINDOW_MS, None)]):
        stairs += [f"{x(ta):.1f},{y(ea):.1f}", f"{x(tb):.1f},{y(ea):.1f}"]
    step_up = " ".join(stairs)
    dots = "\n".join(f'    <circle cx="{x(t):.1f}" cy="{y(e):.1f}" r="2.6"/>' for t, e in points)

    font = 'font-family="Archivo Variable,system-ui,sans-serif"'
    yticks = "\n".join(
        f'    <line x1="66" y1="{y(j):.0f}" x2="70" y2="{y(j):.0f}" stroke="#c9c9c4"/>'
        f'<text x="61" y="{y(j) + 4:.0f}">{j:g}</text>' for j in (0, 1, 2, 3))
    xticks = "\n".join(
        f'    <line x1="{x(ms):.0f}" y1="300" x2="{x(ms):.0f}" y2="304" stroke="#c9c9c4"/>'
        f'<text x="{x(ms):.0f}" y="318">{ms:g}</text>' for ms in (0, 5, 10, 15, 20))
    daemon = "\n".join(
        f'  <line x1="{x(ms):.1f}" y1="50" x2="{x(ms):.1f}" y2="300" stroke="#b93a0a" '
        f'stroke-width="1.2" stroke-dasharray="5 4"/>' for ms in (0, 10, 20))

    print(f'''<svg viewBox="0 0 720 360" role="img" aria-label="Staircase plot of the MI210 energy counter read back to back under load over 22 milliseconds. The value climbs in steps about every 1.2 milliseconds, reaching about 3.3 joules. Dashed vertical lines at 0, 10 and 20 milliseconds mark where a 10 millisecond sampling thread reads it." xmlns="http://www.w3.org/2000/svg">
  <g stroke="#c9c9c4" stroke-width="1">
    <line x1="70" y1="50" x2="70" y2="300"/>
    <line x1="70" y1="300" x2="700" y2="300"/>
  </g>
  <g {font} font-size="12.5" fill="#5f5f5c" text-anchor="end">
{yticks}
  </g>
  <g {font} font-size="12.5" fill="#5f5f5c" text-anchor="middle">
{xticks}
  </g>
  <text x="70" y="36" {font} font-size="13" fill="#5f5f5c">energy since the first read (J)</text>
  <text x="700" y="342" text-anchor="end" {font} font-size="13" fill="#5f5f5c">time (ms)</text>
{daemon}
  <text x="{x(10) + 6:.0f}" y="66" {font} font-size="11.5" fill="#9a3412">read by a 10 ms sampling thread</text>
  <polyline fill="none" stroke="#111111" stroke-width="2" points="{step_up}"/>
  <g fill="#111111">
{dots}
  </g>
  <text x="{x(1):.0f}" y="{y(2.2):.0f}" {font} font-size="11.5" fill="#3d3d3d">a new value about every 1.2 ms</text>
</svg>''')
    return 0


if __name__ == "__main__":
    sys.exit(main())
