
W, H = 1420, 520

# ---------- Restrained Academic Palette ----------
COLORS = {
    "bg": "#FFFFFF",
    "module_fill": "#FAFAFA", "module_stroke": "#C7CBD1",
    "box_fill": "#FFFFFF", "box_stroke": "#4A4F57",
    "box_fill_alt": "#F1F2F4", "box_stroke_alt": "#4A4F57",
    "accent": "#1F3A5F", "box_fill_accent": "#EEF2F7", "box_stroke_accent": "#1F3A5F",
    "output_fill": "#FFFFFF", "output_stroke": "#1F3A5F",
    "text": "#1A1D21", "text_light": "#5B6068", "text_accent": "#1F3A5F",
    "line": "#6B7078",
    "module_title": "#1A1D21"
}

FONT = "Times New Roman, Times, serif"

class SVGBuilder:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.layer_bg, self.layer_modules, self.layer_edges, self.layer_nodes = [], [], [], []

        self.layer_bg.append(f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" font-family="{FONT}">')
        self.layer_bg.append(f'<rect width="{w}" height="{h}" fill="{COLORS["bg"]}"/>')
        self.layer_bg.append(f'''<defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="4.6" markerHeight="4.6" orient="auto-start-reverse">
                <path d="M0,1.2 L8,5 L0,8.8 z" fill="{COLORS["line"]}"/>
            </marker>
            <marker id="arrow-accent" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="4.6" markerHeight="4.6" orient="auto-start-reverse">
                <path d="M0,1.2 L8,5 L0,8.8 z" fill="{COLORS["accent"]}"/>
            </marker>
        </defs>''')

    def rect(self, x, y, w, h, fill, stroke, rx=6, sw=1.3, dash="", layer="nodes"):
        d_attr = f' stroke-dasharray="{dash}"' if dash else ""
        el = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d_attr}/>'
        (self.layer_modules if layer == "modules" else self.layer_nodes).append(el)

    def circle(self, cx, cy, r, fill, stroke, symbol="", sw=1.3):
        self.layer_nodes.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        if symbol:
            self.layer_nodes.append(f'<text x="{cx}" y="{cy}" dy="0.05em" font-size="16" font-family="Arial" text-anchor="middle" fill="{stroke}" dominant-baseline="central">{symbol}</text>')

    def text(self, x, y, lines, size=12, weight="normal", style="normal", anchor="middle", fill=None, lh=18):
        if fill is None: fill = COLORS["text"]
        if isinstance(lines, str): lines = [lines]
        start_y = y - (len(lines) - 1) * lh / 2
        for i, line in enumerate(lines):
            safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            
            # Robust Math Subscript/Superscript logic using baseline-shift (guarantees perfect vertical alignment)
            replacements = {
                "_(uv)": '<tspan baseline-shift="sub" font-size="0.75em">uv</tspan>',
                "_v": '<tspan baseline-shift="sub" font-size="0.75em">v</tspan>',
                "_u": '<tspan baseline-shift="sub" font-size="0.75em">u</tspan>',
                "^(0)": '<tspan baseline-shift="super" font-size="0.75em">(0)</tspan>',
                "^(2)": '<tspan baseline-shift="super" font-size="0.75em">(2)</tspan>',
                "^(l)": '<tspan baseline-shift="super" font-size="0.75em">(l)</tspan>',
                "_{focal}": '<tspan baseline-shift="sub" font-size="0.75em">focal</tspan>',
                "_mlp": '<tspan baseline-shift="sub" font-size="0.75em">mlp</tspan>',
                "_proto": '<tspan baseline-shift="sub" font-size="0.75em">proto</tspan>'
            }
            for k, v in replacements.items():
                safe = safe.replace(k, v)
                
            self.layer_nodes.append(f'<text x="{x}" y="{start_y + i * lh}" font-size="{size}" font-weight="{weight}" font-style="{style}" text-anchor="{anchor}" fill="{fill}" dominant-baseline="central">{safe}</text>')

    def line(self, x1, y1, x2, y2, arrow=True, sw=1.4, stroke=None, accent=False, dash=""):
        stroke = COLORS["accent"] if accent else (stroke or COLORS["line"])
        marker = "arrow-accent" if accent else "arrow"
        a_attr = f' marker-end="url(#{marker})"' if arrow else ""
        d_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.layer_edges.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{a_attr}{d_attr}/>')

    def route(self, x1, y1, x2, y2, dir="h", r=10, arrow=True, sw=1.4, stroke=None, accent=False, dash=""):
        stroke = COLORS["accent"] if accent else (stroke or COLORS["line"])
        marker = "arrow-accent" if accent else "arrow"
        a_attr = f' marker-end="url(#{marker})"' if arrow else ""
        d_attr = f' stroke-dasharray="{dash}"' if dash else ""
        
        if dir == "h":
            dx = 1 if x2 > x1 else -1
            dy = 1 if y2 > y1 else -1
            if abs(x2 - x1) < r or abs(y2 - y1) < r: r = min(abs(x2 - x1), abs(y2 - y1))
            d = f"M {x1} {y1} L {x2 - dx*r} {y1} Q {x2} {y1} {x2} {y1 + dy*r} L {x2} {y2}"
        else: # "v"
            dx = 1 if x2 > x1 else -1
            dy = 1 if y2 > y1 else -1
            if abs(x2 - x1) < r or abs(y2 - y1) < r: r = min(abs(x2 - x1), abs(y2 - y1))
            d = f"M {x1} {y1} L {x1} {y2 - dy*r} Q {x1} {y2} {x1 + dx*r} {y2} L {x2} {y2}"

        self.layer_edges.append(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}"{a_attr}{d_attr}/>')

    def custom_path(self, d, arrow=True, sw=1.4, stroke=None, accent=False, dash=""):
        stroke = COLORS["accent"] if accent else (stroke or COLORS["line"])
        marker = "arrow-accent" if accent else "arrow"
        a_attr = f' marker-end="url(#{marker})"' if arrow else ""
        d_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.layer_edges.append(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}"{a_attr}{d_attr}/>')

    def save(self, filename):
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(self.layer_bg + self.layer_modules + self.layer_edges + self.layer_nodes + ['</svg>']))

svg = SVGBuilder(W, H)

# ==================================================================
# Module Backgrounds & Headers
# ==================================================================
zones = [
    (10, 290, "Graph Construction & Encoding"),
    (310, 410, "Structural Module"),
    (730, 460, "Dual-Head Classifier"),
    (1200, 210, "Optimization")
]

for x, w, title in zones:
    svg.rect(x, 20, w, 480, COLORS["module_fill"], COLORS["module_stroke"], dash="5 4", layer="modules")
    svg.text(x + w/2, 45, title, size=13.5, weight="bold", fill=COLORS["module_title"])
    svg.layer_modules.append(f'<line x1="{x+10}" y1="65" x2="{x+w-10}" y2="65" stroke="{COLORS["module_stroke"]}" stroke-width="1"/>')

# ==================================================================
# Module 1: Graph Construction & Encoding
# ==================================================================
# Y centers perfectly aligned to the concat block center (y=202.5)
m1_y = [97.5, 167.5, 237.5, 307.5]
labels = [["Tweets (768-d)", "RoBERTa"], ["Description (768-d)", "RoBERTa"], ["Numeric (5-d)"], ["Categorical (3-d)"]]

for i, y in enumerate(m1_y):
    svg.rect(30, y - 20, 120, 40, COLORS["box_fill"], COLORS["box_stroke"])
    svg.text(90, y, labels[i], size=11, lh=15)
    svg.route(150, y, 185, 202.5, dir="h", arrow=False)

# Concat Box
svg.rect(185, 92.5, 95, 220, COLORS["box_fill_accent"], COLORS["box_stroke_accent"])
svg.text(232.5, 202.5, ["Concat", "&", "Project"], size=13, weight="bold", fill=COLORS["text_accent"])

# Edge Features Box
svg.rect(30, 400, 250, 55, COLORS["box_fill_alt"], COLORS["box_stroke_alt"])
svg.text(155, 427.5, ["Edge Features e_(uv) ∈ ℝ³", "(Cosine Similarity + Relation Type)"], size=11.5, weight="bold")

# Output M1 -> M2
svg.line(280, 202.5, 330, 202.5, accent=True)
svg.text(305, 185, "h_v^(0)", size=13, weight="bold", fill=COLORS["text_accent"])

# ==================================================================
# Module 2: Structural Module (Aligned vertically)
# ==================================================================
# Stacked Card Effect
svg.rect(350, 82.5, 340, 360, COLORS["box_fill_alt"], COLORS["box_stroke"], sw=1.1)
svg.rect(340, 92.5, 340, 360, "#Fcfcfc", COLORS["box_stroke"], sw=1.1)
svg.rect(330, 102.5, 340, 360, COLORS["box_fill"], COLORS["box_stroke"])

svg.text(500, 130, "TransformerConv Layer", size=13.5, weight="bold")
svg.text(645, 130, "× L", size=13, weight="bold", fill=COLORS["text_light"], anchor="start")

# Internal Block
svg.rect(360, 170, 280, 65, COLORS["box_fill_accent"], COLORS["box_stroke_accent"])
svg.text(500, 202.5, ["Edge-Modulated Attention", "α_(uv) = softmax(W_q h_u · W_k h_v + W_e e_(uv))"], size=11, fill=COLORS["text_accent"])

svg.rect(425, 265, 150, 45, COLORS["box_fill"], COLORS["box_stroke"])
svg.text(500, 287.5, "Aggregate & Add", size=12)

svg.circle(500, 345, 12, COLORS["bg"], COLORS["box_stroke"])
svg.text(500, 343.5, "+", size=18)

svg.rect(425, 380, 150, 45, COLORS["box_fill"], COLORS["box_stroke"])
svg.text(500, 402.5, "Residual + LayerNorm", size=12)

# Pipeline Routing
svg.line(500, 235, 500, 265)
svg.line(500, 310, 500, 333)
svg.line(500, 357, 500, 380)

# Edge Feature Routing (Snakes efficiently into the attention box)
svg.line(280, 430, 330, 430, accent=True)
svg.text(300, 410, "e_(uv)", size=13, weight="bold", fill=COLORS["text_accent"], anchor="start")

# Skip Connection (Hidden cleanly inside the card)
svg.custom_path("M 345 202.5 L 345 392.5 Q 345 402.5 355 402.5 L 425 402.5", arrow=True)
svg.text(355, 385, "skip", size=11, style="italic", fill=COLORS["text_light"])

# Output M2 -> M3
svg.route(670, 202.5, 750, 145, dir="h", arrow=False)
svg.text(715, 185, "h_v^(2)", size=13, weight="bold", fill=COLORS["text_accent"])

# ==================================================================
# Module 3: Dual-Head Classifier (Perfect Alignment)
# ==================================================================
# Distribution Bus
svg.line(750, 145, 1120, 145, arrow=False)
svg.line(800, 145, 800, 180)
svg.line(960, 145, 960, 180)
svg.line(1120, 145, 1120, 180)

# Classifier Heads
svg.rect(740, 180, 120, 65, COLORS["box_fill"], COLORS["box_stroke"])
svg.text(800, 212.5, ["MLP Head", "z_mlp ∈ ℝ²"], size=12, weight="bold")

svg.rect(900, 180, 120, 65, COLORS["box_fill_accent"], COLORS["box_stroke_accent"])
svg.text(960, 212.5, ["Gate Network", "γ ∈ (0, 1)"], size=12, weight="bold", fill=COLORS["text_accent"])

svg.rect(1060, 180, 120, 65, COLORS["box_fill"], COLORS["box_stroke"])
svg.text(1120, 212.5, ["Prototype Head", "cos(h, C) × τ"], size=12, weight="bold")

# Unified Weighted Combination Box
svg.rect(740, 320, 440, 60, COLORS["box_fill_alt"], COLORS["box_stroke_alt"])
svg.text(960, 350, ["Weighted Combination", "z_v = γ · z_mlp + (1 - γ) · z_proto"], size=13, weight="bold")

# Route directly into the combination box
svg.line(800, 245, 800, 320)  # MLP down
svg.line(1120, 245, 1120, 320) # Proto down
svg.line(960, 245, 960, 320, dash="5 4") # Gate down (dashed scalar weight)

# Output M3 -> M4
svg.line(1180, 350, 1240, 350, accent=True)
svg.text(1210, 335, "z_v", size=14, weight="bold", fill=COLORS["text_accent"])

# ==================================================================
# Module 4: Optimization
# ==================================================================
svg.rect(1240, 175, 140, 60, COLORS["output_fill"], COLORS["output_stroke"], rx=30)
svg.text(1310, 205, ["Bot / Human", "Prediction"], size=13, weight="bold", fill=COLORS["text_accent"])

svg.rect(1240, 320, 140, 60, COLORS["box_fill_alt"], COLORS["box_stroke_alt"])
svg.text(1310, 350, ["Focal Loss", "ℒ_{focal}"], size=13, weight="bold")

# Final connection
svg.line(1310, 235, 1310, 320)

# ==================================================================
svg.save("architecture.svg")
print("Saved AAAI publication-ready diagram with perfect math alignments to architecture.svg")
