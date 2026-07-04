import math

W, H = 480, 280

# ---------- AAAI Academic Palette ----------
COLORS = {
    "bg": "#FFFFFF",
    "box_blue": "#E7F5FF", "stroke_blue": "#339AF0",
    "box_red": "#FFF5F5", "stroke_red": "#FA5252",
    "box_green": "#EBFBEE", "stroke_green": "#40C057",
    "text_main": "#212529",
    "text_muted": "#6C757D",
    "line": "#DEE2E6",
    "line_dark": "#495057",
}

FONT = "Times New Roman, Times, serif"

class SVGBuilder:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.layer_bg = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" font-family="{FONT}">',
                         f'<rect width="{w}" height="{h}" fill="{COLORS["bg"]}"/>']
        self.layer_edges = []
        self.layer_nodes = []
        
        # Define multiple arrow heads for the different edge weights
        self.layer_bg.append(f'''<defs>
            <filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">
                <feDropShadow dx="1.5" dy="1.5" stdDeviation="1.5" flood-opacity="0.12"/>
            </filter>
            <marker id="arrow_dark" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M0,0 L10,5 L0,10 z" fill="{COLORS["line_dark"]}"/>
            </marker>
            <marker id="arrow_red" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M0,0 L10,5 L0,10 z" fill="{COLORS["stroke_red"]}"/>
            </marker>
            <marker id="arrow_green" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M0,0 L10,5 L0,10 z" fill="{COLORS["stroke_green"]}"/>
            </marker>
        </defs>''')

    def text(self, x, y, lines, size=12, weight="normal", style="normal", anchor="middle", fill=COLORS["text_main"], lh=14):
        if isinstance(lines, str): lines = [lines]
        start_y = y - (len(lines) - 1) * lh / 2
        for i, line in enumerate(lines):
            safe_line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            self.layer_nodes.append(f'<text x="{x}" y="{start_y + i * lh}" font-size="{size}" font-weight="{weight}" font-style="{style}" text-anchor="{anchor}" fill="{fill}" dominant-baseline="central">{safe_line}</text>')

    def line(self, x1, y1, x2, y2, sw=1.5, dash="", stroke=COLORS["line_dark"]):
        d_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.layer_bg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{d_attr}/>')

    def connect(self, x1, y1, x2, y2, r1=12, r2=22, sw=1.5, dash="", stroke=COLORS["line_dark"], marker="arrow_dark", label=None):
        """Draws a line perfectly between the edges of two circles, calculating the trigonometry automatically."""
        d = math.hypot(x2 - x1, y2 - y1)
        if d == 0: return
        nx, ny = (x2 - x1) / d, (y2 - y1) / d
        
        sx, sy = x1 + nx * r1, y1 + ny * r1
        ex, ey = x2 - nx * r2, y2 - ny * r2

        d_attr = f' stroke-dasharray="{dash}"' if dash else ""
        a_attr = f' marker-end="url(#{marker})"'
        
        self.layer_edges.append(f'<line x1="{sx}" y1="{sy}" x2="{ex}" y2="{ey}" stroke="{stroke}" stroke-width="{sw}"{d_attr}{a_attr}/>')

        if label:
            mx, my = (sx + ex) / 2, (sy + ey) / 2
            # Add a small white background pill behind the edge label
            self.layer_edges.append(f'<rect x="{mx-18}" y="{my-8}" width="36" height="16" fill="{COLORS["bg"]}" opacity="0.85" rx="3"/>')
            self.text(mx, my, label, size=10, fill=stroke, weight="bold")

    def circle(self, cx, cy, r, fill, stroke, text_str="", shadow=False):
        s_attr = ' filter="url(#shadow)"' if shadow else ""
        self.layer_nodes.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"{s_attr}/>')
        if text_str:
            self.layer_nodes.append(f'<text x="{cx}" y="{cy}" dy="0.1em" font-size="11" font-weight="bold" text-anchor="middle" dominant-baseline="central" fill="{stroke}">{text_str}</text>')

    def save(self, filename):
        all_parts = self.layer_bg + self.layer_edges + self.layer_nodes + ['</svg>']
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(all_parts))

# ==================================================================
# Initialize Builder
# ==================================================================
svg = SVGBuilder(W, H)

# Center vertical divider
svg.line(W/2, 20, W/2, H - 20, sw=1, dash="4 4", stroke=COLORS["line"])

# ==================================================================
# Left Pane: Without Edge Features
# ==================================================================
svg.text(W/4, 25, "Without Edge Features", size=14, weight="bold")
svg.text(W/4, 42, "All neighbors treated equally", size=11, style="italic", fill=COLORS["text_muted"])

# Coordinates for Left Pane
center_L = (120, 160)
nodes_L = {
    "h1": (50, 100),  # Human Top-Left
    "h2": (50, 220),  # Human Bottom-Left
    "b1": (190, 100), # Bot Top-Right
    "b2": (190, 220), # Bot Bottom-Right
}

# Standard equal-weight connections
for key, pos in nodes_L.items():
    svg.connect(pos[0], pos[1], center_L[0], center_L[1], r1=12, r2=22, sw=1.5, stroke=COLORS["line_dark"], marker="arrow_dark", label="α")

# Draw Nodes Left
svg.circle(center_L[0], center_L[1], 18, COLORS["box_red"], COLORS["stroke_red"], "Bot (v)", shadow=True)
svg.circle(nodes_L["h1"][0], nodes_L["h1"][1], 12, COLORS["box_blue"], COLORS["stroke_blue"], "H", shadow=True)
svg.circle(nodes_L["h2"][0], nodes_L["h2"][1], 12, COLORS["box_blue"], COLORS["stroke_blue"], "H", shadow=True)
svg.circle(nodes_L["b1"][0], nodes_L["b1"][1], 12, COLORS["box_red"], COLORS["stroke_red"], "B", shadow=True)
svg.circle(nodes_L["b2"][0], nodes_L["b2"][1], 12, COLORS["box_red"], COLORS["stroke_red"], "B", shadow=True)


# ==================================================================
# Right Pane: With Edge Features (AdaRelBot)
# ==================================================================
svg.text(3 * W/4, 25, "With Edge Features (AdaRelBot)", size=14, weight="bold")
svg.text(3 * W/4, 42, "Dissimilar edges are attenuated", size=11, style="italic", fill=COLORS["text_muted"])

# Coordinates for Right Pane
center_R = (360, 160)
nodes_R = {
    "h1": (290, 100), # Human Top-Left
    "h2": (290, 220), # Human Bottom-Left
    "b1": (430, 100), # Bot Top-Right
    "b2": (430, 220), # Bot Bottom-Right
}

# Modulated connections
# Human -> Bot (Heterophilous: Weak)
svg.connect(nodes_R["h1"][0], nodes_R["h1"][1], center_R[0], center_R[1], r1=12, r2=22, 
            sw=1.2, dash="4 3", stroke=COLORS["stroke_red"], marker="arrow_red", label="cos ≈ 0.1")
svg.connect(nodes_R["h2"][0], nodes_R["h2"][1], center_R[0], center_R[1], r1=12, r2=22, 
            sw=1.2, dash="4 3", stroke=COLORS["stroke_red"], marker="arrow_red", label="cos ≈ 0.1")

# Bot -> Bot (Homophilous: Strong)
svg.connect(nodes_R["b1"][0], nodes_R["b1"][1], center_R[0], center_R[1], r1=12, r2=24, 
            sw=3.5, stroke=COLORS["stroke_green"], marker="arrow_green", label="cos ≈ 0.9")
svg.connect(nodes_R["b2"][0], nodes_R["b2"][1], center_R[0], center_R[1], r1=12, r2=24, 
            sw=3.5, stroke=COLORS["stroke_green"], marker="arrow_green", label="cos ≈ 0.9")

# Draw Nodes Right
svg.circle(center_R[0], center_R[1], 18, COLORS["box_red"], COLORS["stroke_red"], "Bot (v)", shadow=True)
svg.circle(nodes_R["h1"][0], nodes_R["h1"][1], 12, COLORS["box_blue"], COLORS["stroke_blue"], "H", shadow=True)
svg.circle(nodes_R["h2"][0], nodes_R["h2"][1], 12, COLORS["box_blue"], COLORS["stroke_blue"], "H", shadow=True)
svg.circle(nodes_R["b1"][0], nodes_R["b1"][1], 12, COLORS["box_red"], COLORS["stroke_red"], "B", shadow=True)
svg.circle(nodes_R["b2"][0], nodes_R["b2"][1], 12, COLORS["box_red"], COLORS["stroke_red"], "B", shadow=True)

# ==================================================================
# Save Output
# ==================================================================
svg.save("message_passing.svg")
print("Saved clean, single-column message passing graphic to message_passing.svg")
