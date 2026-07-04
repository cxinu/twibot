import math

W, H = 480, 280

# ---------- AAAI Academic Palette ----------
COLORS = {
    "bg": "#FFFFFF",
    "box_blue": "#E7F5FF", "stroke_blue": "#339AF0",
    "box_red": "#FFF5F5", "stroke_red": "#FA5252",
    "text_main": "#212529",
    "text_muted": "#495057",
    "line": "#ADB5BD",
    "line_dark": "#495057"
}

FONT = "Times New Roman, Times, serif"

class SVGBuilder:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.layer_bg = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" font-family="{FONT}">',
                         f'<rect width="{w}" height="{h}" fill="{COLORS["bg"]}"/>']
        self.layer_edges = []
        self.layer_nodes = []
        
        self.layer_bg.append('''<defs>
            <filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">
                <feDropShadow dx="1.5" dy="1.5" stdDeviation="1.5" flood-opacity="0.12"/>
            </filter>
        </defs>''')

    def text(self, x, y, lines, size=12, weight="normal", style="normal", anchor="middle", fill=COLORS["text_main"], lh=14):
        if isinstance(lines, str): lines = [lines]
        start_y = y - (len(lines) - 1) * lh / 2
        for i, line in enumerate(lines):
            safe_line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            self.layer_nodes.append(f'<text x="{x}" y="{start_y + i * lh}" font-size="{size}" font-weight="{weight}" font-style="{style}" text-anchor="{anchor}" fill="{fill}" dominant-baseline="central">{safe_line}</text>')

    def line(self, x1, y1, x2, y2, sw=1.5, dash="", stroke=COLORS["line_dark"]):
        d_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.layer_edges.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{d_attr}/>')

    def circle(self, cx, cy, r, fill, stroke, text_str="", shadow=False):
        s_attr = ' filter="url(#shadow)"' if shadow else ""
        self.layer_nodes.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"{s_attr}/>')
        if text_str:
            # dy="0.1em" helps perfectly center the text vertically in most PDF/SVG renderers
            self.layer_nodes.append(f'<text x="{cx}" y="{cy}" dy="0.1em" font-size="11" font-weight="bold" text-anchor="middle" dominant-baseline="central" fill="{stroke}">{text_str}</text>')

    def save(self, filename):
        all_parts = self.layer_bg + self.layer_edges + self.layer_nodes + ['</svg>']
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(all_parts))

# ==================================================================
# Initialize Builder
# ==================================================================
svg = SVGBuilder(W, H)

# Divider
svg.line(W/2, 20, W/2, H - 40, sw=1, dash="4 4", stroke=COLORS["line"])

# ==================================================================
# Left Pane: Homophily (Ideal)
# ==================================================================
svg.text(W/4, 25, "Homophily (Ideal)", size=14, weight="bold")

# Human Cluster (Left)
h_nodes = [(80, 120), (45, 90), (50, 150), (110, 85), (115, 145)]
# Bot Cluster (Right)
b_nodes = [(175, 120), (145, 85), (140, 145), (210, 90), (205, 150)]

# Intra-cluster edges (Homophilous - Solid Gray)
for p in h_nodes[1:]:
    svg.line(h_nodes[0][0], h_nodes[0][1], p[0], p[1], sw=1.5, stroke=COLORS["line"])
svg.line(h_nodes[3][0], h_nodes[3][1], h_nodes[1][0], h_nodes[1][1], sw=1.5, stroke=COLORS["line"])

for p in b_nodes[1:]:
    svg.line(b_nodes[0][0], b_nodes[0][1], p[0], p[1], sw=1.5, stroke=COLORS["line"])
svg.line(b_nodes[4][0], b_nodes[4][1], b_nodes[2][0], b_nodes[2][1], sw=1.5, stroke=COLORS["line"])

# Inter-cluster edge (Heterophilous anomaly - Dashed Red)
svg.line(h_nodes[4][0], h_nodes[4][1], b_nodes[2][0], b_nodes[2][1], sw=1.5, dash="4 3", stroke=COLORS["stroke_red"])

# Draw Nodes
for p in h_nodes:
    svg.circle(p[0], p[1], 11, COLORS["box_blue"], COLORS["stroke_blue"], "H", shadow=True)
for p in b_nodes:
    svg.circle(p[0], p[1], 11, COLORS["box_red"], COLORS["stroke_red"], "B", shadow=True)

# Captions
svg.text(W/4, 200, "Birds of a feather cluster together.", size=11, style="italic", fill=COLORS["text_muted"])
svg.text(W/4, 218, "Standard GNNs thrive.", size=12, weight="bold")


# ==================================================================
# Right Pane: Heterophily (Real-World)
# ==================================================================
svg.text(3 * W/4, 25, "Heterophily (Real-World)", size=14, weight="bold")

center = (360, 120)
radius = 45

# Generate 6 surrounding nodes
surrounding = []
for i in range(6):
    angle = math.radians(i * 60 - 30)
    surrounding.append((center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle)))

# Connect central bot to neighbors (Heterophilous - Dashed Red)
for i, p in enumerate(surrounding):
    if i == 4: # One bot neighbor to show reality
        svg.line(center[0], center[1], p[0], p[1], sw=1.5, stroke=COLORS["line"])
    else:
        svg.line(center[0], center[1], p[0], p[1], sw=1.5, dash="4 3", stroke=COLORS["stroke_red"])

# Add some cross-links between humans to show organic structure
svg.line(surrounding[0][0], surrounding[0][1], surrounding[5][0], surrounding[5][1], sw=1.5, stroke=COLORS["line"])
svg.line(surrounding[1][0], surrounding[1][1], surrounding[2][0], surrounding[2][1], sw=1.5, stroke=COLORS["line"])

# Draw Nodes
svg.circle(center[0], center[1], 12, COLORS["box_red"], COLORS["stroke_red"], "B", shadow=True) # Central Bot
for i, p in enumerate(surrounding):
    if i == 4:
        svg.circle(p[0], p[1], 11, COLORS["box_red"], COLORS["stroke_red"], "B", shadow=True)
    else:
        svg.circle(p[0], p[1], 11, COLORS["box_blue"], COLORS["stroke_blue"], "H", shadow=True)

# Captions
svg.text(3 * W/4, 200, "Bots hide by following humans.", size=11, style="italic", fill=COLORS["text_muted"])
svg.text(3 * W/4, 218, "Standard GNNs get confused.", size=12, weight="bold")


# ==================================================================
# Legend (Bottom Center)
# ==================================================================
leg_y = 260
svg.circle(W/2 - 90, leg_y, 8, COLORS["box_blue"], COLORS["stroke_blue"], "")
svg.text(W/2 - 75, leg_y, "Human", size=11, anchor="start")

svg.circle(W/2 - 20, leg_y, 8, COLORS["box_red"], COLORS["stroke_red"], "")
svg.text(W/2 - 5, leg_y, "Bot", size=11, anchor="start")

svg.line(W/2 + 40, leg_y, W/2 + 65, leg_y, sw=1.5, dash="4 3", stroke=COLORS["stroke_red"])
svg.text(W/2 + 70, leg_y, "Heterophilous Edge", size=11, anchor="start")

# ==================================================================
# Save Output
# ==================================================================
svg.save("homophily_heterophily.svg")
print("Saved clean, single-column academic graphic to homophily_heterophily.svg")
