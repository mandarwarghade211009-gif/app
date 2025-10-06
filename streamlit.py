#!/usr/bin/env python3
# Figma to Angular Converter - Professional Multi-Agent Streamlit Application

import os
import io
import json
import datetime
import shutil
import time
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set

import requests
import streamlit as st

# ========= MUST be first Streamlit call =========
st.set_page_config(
    page_title="Figma to Angular Converter",
    page_icon="ℹ️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# # ========= Enhanced Professional Light Theme CSS =========
LIGHT_THEME_CSS = """
<style>
:root {
  --bg-primary: #ffffff;
  --bg-secondary: #f8fafc;
  --bg-tertiary: #f1f5f9;
  --text-primary: #0f172a;
  --text-secondary: #475569;
  --text-muted: #64748b;
  --border: #e2e8f0;
  --accent-blue: #3b82f6;
  --accent-green: #10b981;
  --accent-purple: #8b5cf6;
  --accent-amber: #f59e0b;
  --accent-red: #ef4444;
  --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
}

.stApp { 
  background-color: var(--bg-primary); 
  color: var(--text-primary);
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif; 
}

section.main > div { 
  background-color: var(--bg-primary); 
  padding: 1.5rem; 
}

h1, h2, h3, h4, h5, h6 { 
  color: var(--text-primary); 
  font-weight: 700; 
  letter-spacing: -0.025em; 
}

h1 { font-size: 2.5rem; margin-bottom: 1rem; }
h2 { font-size: 2rem; margin-top: 2rem; margin-bottom: 1rem; }
h3 { font-size: 1.75rem; margin-top: 1.5rem; margin-bottom: 0.75rem; }
h4 { font-size: 1.5rem; margin-top: 1rem; margin-bottom: 0.5rem; }

.stButton > button {
  background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-purple) 100%);
  color: white;
  border: none;
  border-radius: 10px;
  padding: 0.75rem 1.5rem;
  font-weight: 600;
  font-size: 1rem;
  box-shadow: 0 4px 6px -1px rgba(59,130,246,0.3);
  transition: all 0.3s ease;
  cursor: pointer;
}

.stButton > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 12px -2px rgba(59,130,246,0.4);
}

.stButton > button:disabled {
  background: linear-gradient(135deg, #94a3b8 0%, #cbd5e1 100%);
  cursor: not-allowed;
  opacity: 0.6;
}

.stDownloadButton > button {
  background-color: var(--accent-green);
  color: white;
  border: none;
  border-radius: 8px;
  padding: 0.625rem 1.25rem;
  font-weight: 600;
  box-shadow: 0 2px 4px rgba(16,185,129,0.2);
  transition: all 0.2s ease;
}

.stTextInput > div > div > input {
  border: 2px solid var(--border);
  border-radius: 8px;
  padding: 0.625rem 0.875rem;
  font-size: 0.9375rem;
  transition: all 0.2s ease;
}

.stTextInput > div > div > input:focus {
  border-color: var(--accent-blue);
  box-shadow: 0 0 0 3px rgba(59,130,246,0.1);
}

.stTabs [data-baseweb="tab-list"] {
  gap: 0.5rem;
  background-color: var(--bg-secondary);
  padding: 0.5rem;
  border-radius: 12px;
  box-shadow: var(--shadow);
}

.stTabs [data-baseweb="tab"] {
  background-color: transparent;
  border-radius: 8px;
  color: var(--text-secondary);
  font-weight: 600;
  padding: 0.75rem 1.5rem;
}

.stTabs [aria-selected="true"] {
  background-color: white;
  color: var(--accent-blue);
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

section[data-testid="stSidebar"] {
  background-color: var(--bg-secondary);
  border-right: 1px solid var(--border);
}

.stAlert {
  border-radius: 10px;
  border: 1px solid var(--border);
  padding: 1rem;
  margin: 1rem 0;
}

.stSuccess {
  background-color: #f0fdf4;
  border-color: var(--accent-green);
  color: #166534;
}

.stError {
  background-color: #fef2f2;
  border-color: var(--accent-red);
  color: #991b1b;
}

.stInfo {
  background-color: #eff6ff;
  border-color: var(--accent-blue);
  color: #1e40af;
}

.log-container {
  background-color: var(--bg-tertiary);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem;
  max-height: 400px;
  overflow-y: auto;
  font-family: 'Courier New', monospace;
  font-size: 0.875rem;
}

hr { border-color: var(--border); margin: 1.5rem 0; }

@media (max-width: 768px) {
  .stButton > button {
    padding: 0.5rem 1rem;
    font-size: 0.875rem;
  }
}
</style>
"""
st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)

# ========= PDF Libraries =========
USE_REPORTLAB = False
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    USE_REPORTLAB = True
except Exception:
    pass

FPDF_AVAILABLE = False
if not USE_REPORTLAB:
    try:
        from fpdf import FPDF
        FPDF_AVAILABLE = True
    except Exception:
        pass

# ========= CrewAI imports =========
CREW_AVAILABLE = True
try:
    from crewai import Agent, Task, Crew, Process, LLM
except Exception:
    CREW_AVAILABLE = False

# ========= Configuration =========
APP_TITLE = "Figma to Angular Converter"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDINVS-5WrytukofOLKEdufBAGnE5u_qSI")
LLM_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini/gemini-2.0-flash")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "64000"))
DEFAULT_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "600"))
MAX_RETRIES = 3
RENDER_BATCH_SIZE = 200
RENDER_FORMAT = "svg"

# ========= Session State Defaults =========
defaults = {
    "small_metadata": None,
    "large_metadata": None,
    "large_chunk1_json": None,
    "large_chunk2_json": None,
    "angular_small": None,
    "angular_chunk1": None,
    "angular_chunk2": None,
    "angular_merged_app": None,
    "logs_small": [],
    "logs_large": [],
    "chunk1_done": False,
    "chunk2_done": False,
    "metadata_extracted_small": False,
    "metadata_extracted_large": False,
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ========= Validation Functions =========
def validate_input(value: str, field_name: str) -> Tuple[bool, str]:
    """Validate single input field"""
    if not value:
        return False, f"{field_name} cannot be empty"
    if value != value.strip():
        return False, f"{field_name} contains leading/trailing whitespace"
    if len(value.strip()) == 0:
        return False, f"{field_name} cannot be only whitespace"
    return True, ""

def validate_figma_inputs(file_key: str, node_ids: str, token: str) -> Tuple[bool, List[str]]:
    """Validate all Figma input fields"""
    errors = []
    
    valid, msg = validate_input(file_key, "File Key")
    if not valid:
        errors.append(msg)
    
    valid, msg = validate_input(node_ids, "Node IDs")
    if not valid:
        errors.append(msg)
    
    valid, msg = validate_input(token, "Access Token")
    if not valid:
        errors.append(msg)
    
    return len(errors) == 0, errors

# ========= Logging Functions =========
def _ts() -> str:
    """Get current timestamp"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str, option: str = "small", level: str = "INFO") -> None:
    """Add log entry"""
    entry = {"timestamp": _ts(), "message": msg, "level": level}
    if option == "small":
        st.session_state.logs_small.append(entry)
    else:
        st.session_state.logs_large.append(entry)
    print(f"[{entry['timestamp']}] [{level}] [{option.upper()}] {msg}")

def clear_logs(option: str) -> None:
    """Clear logs for specific option"""
    if option == "small":
        st.session_state.logs_small = []
    else:
        st.session_state.logs_large = []

def render_logs(option: str) -> None:
    """Render execution logs with enhanced styling"""
    logs = st.session_state.logs_small if option == "small" else st.session_state.logs_large
    
    if not logs:
        st.info("No logs available. Logs will appear here during processing.")
        return
    
    log_html = '<div class="log-container">'
    for entry in logs[-100:]:
        level = entry.get("level", "INFO")
        ts = entry.get("timestamp", "")
        msg = entry.get("message", "")
        
        icons = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "WARNING": "⚠️",
            "ERROR": "❌"
        }
        icon = icons.get(level, "📄")
        
        colors = {
            "INFO": "#3b82f6",
            "SUCCESS": "#10b981",
            "WARNING": "#f59e0b",
            "ERROR": "#ef4444"
        }
        color = colors.get(level, "#64748b")
        
        log_html += f'<div style="margin: 0.25rem 0; color: {color};">{icon} <strong>[{ts}]</strong> {msg}</div>'
    
    log_html += '</div>'
    st.markdown(log_html, unsafe_allow_html=True)

# ========= Directory Management =========
OUTPUT_DIR = Path("output")
SMALL_DIR = OUTPUT_DIR / "small"
LARGE_DIR = OUTPUT_DIR / "large"
CHUNKS_DIR = LARGE_DIR / "chunks"
CHUNK1_DIR = CHUNKS_DIR / "chunk1"
CHUNK2_DIR = CHUNKS_DIR / "chunk2"

def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist"""
    path.mkdir(parents=True, exist_ok=True)

def clean_dir(path: Path, option: str) -> None:
    """Clean directory contents while preserving the directory"""
    ensure_dir(path)
    deleted_count = 0
    for item in path.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            deleted_count += 1
        except Exception as e:
            log(f"Failed to delete {item.name}: {str(e)}", option, "WARNING")
    
    if deleted_count > 0:
        log(f"Cleaned {deleted_count} items from {path.name}/", option, "SUCCESS")

def prepare_directories_small(option: str = "small") -> None:
    """Prepare directories for small file processing"""
    log("Preparing output directories...", option, "INFO")
    ensure_dir(OUTPUT_DIR)
    clean_dir(SMALL_DIR, option)
    log("Small file directories ready", option, "SUCCESS")

def prepare_directories_large(option: str = "large") -> None:
    """Prepare directories for large file processing"""
    log("Preparing output directories...", option, "INFO")
    ensure_dir(OUTPUT_DIR)
    clean_dir(LARGE_DIR, option)
    ensure_dir(CHUNKS_DIR)
    clean_dir(CHUNKS_DIR, option)
    ensure_dir(CHUNK1_DIR)
    ensure_dir(CHUNK2_DIR)
    log("Large file directories ready", option, "SUCCESS")

# ========= PDF Generation Functions =========
def build_pdf_reportlab(title: str, body: str) -> bytes:
    """Build PDF using ReportLab"""
    try:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=letter)
        styles = getSampleStyleSheet()
        story = [
            Paragraph(title, styles['Heading1']),
            Spacer(1, 0.2 * inch)
        ]
        
        for line in body.splitlines()[:500]:
            try:
                story.append(Preformatted(line, styles['Code']))
            except Exception:
                try:
                    story.append(Paragraph(line, styles['BodyText']))
                except Exception:
                    pass
        
        doc.build(story)
        return buf.getvalue()
    except Exception:
        return b""

def build_pdf_fpdf(title: str, body: str) -> bytes:
    """Build PDF using FPDF"""
    if not FPDF_AVAILABLE:
        
        return b""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, title, ln=True)
        pdf.ln(5)
        pdf.set_font("Arial", "", 9)
        
        for line in body.splitlines()[:1000]:
            try:
                pdf.multi_cell(0, 4, line)
            except Exception:
                pdf.multi_cell(0, 4, "[encoding error]")
        
        return pdf.output(dest="S").encode("latin-1")
    except Exception:
        return b""

def build_pdf(title: str, body: str) -> bytes:
    """Build PDF using available library"""
    if USE_REPORTLAB:
        blob = build_pdf_reportlab(title, body)
        if blob:
            return blob
    return build_pdf_fpdf(title, body)

# ========= Figma Utility Functions =========
def build_headers(token: str) -> Dict[str, str]:
    """Build Figma API headers"""
    return {
        "Accept": "application/json",
        "X-Figma-Token": token
    }


def chunked(lst: List[str], n: int):
    """Split list into chunks of size n"""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def to_rgba(color: Dict[str, Any]) -> str:
    """Convert Figma color dict to CSS rgba string"""
    try:
        r = int(float(color.get('r', 0)) * 255)
        g = int(float(color.get('g', 0)) * 255)
        b = int(float(color.get('b', 0)) * 255)
        a = float(color.get('a', color.get('opacity', 1)))
        return f"rgba({r}, {g}, {b}, {a})"
    except Exception:
        return "rgba(0,0,0,1)"

def is_nonempty_list(v: Any) -> bool:
    """Check if value is a non-empty list"""
    return isinstance(v, list) and len(v) > 0

# ========= Figma Extraction Functions =========
def fetch_figma_nodes(file_key: str, node_ids: str, token: str, option: str) -> Dict[str, Any]:
    """Fetch nodes from Figma API with retry logic"""
    headers = build_headers(token)
    url = f"https://api.figma.com/v1/files/{file_key}/nodes"
    params = {"ids": node_ids}
    
    for attempt in range(MAX_RETRIES):
        try:
            log(f"Fetching Figma nodes (attempt {attempt + 1}/{MAX_RETRIES})...", option, "INFO")
            resp = requests.get(url, headers=headers, params=params, timeout=60)
            
            if resp.ok:
                nodes_data = resp.json()
                log(f"Successfully fetched Figma metadata ({len(str(nodes_data))} bytes)", option, "SUCCESS")
                return nodes_data
            
            if resp.status_code == 429:
                wait_s = 2 ** attempt
                log(f"Rate limited. Waiting {wait_s}s before retry...", option, "WARNING")
                time.sleep(wait_s)
                continue
            
            error_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
            log(error_msg, option, "ERROR")
            raise RuntimeError(error_msg)
            
        except requests.exceptions.Timeout:
            log(f"Request timeout (attempt {attempt + 1})", option, "WARNING")
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError("Request timeout after all retries")
        except Exception as e:
            log(f"Request error: {str(e)}", option, "ERROR")
            if attempt == MAX_RETRIES - 1:
                raise
    
    raise RuntimeError("Failed to fetch Figma data after all retries")

def walk_nodes_collect_images_and_ids(nodes_payload: dict, option: str) -> Tuple[Set[str], List[str], Dict[str, Dict[str, str]]]:
    """Walk through nodes and collect image references, node IDs, and metadata"""
    log("Collecting image references and node IDs...", option, "INFO")
    
    image_refs: Set[str] = set()
    node_ids: List[str] = []
    node_meta: Dict[str, Dict[str, str]] = {}

    def visit(n: dict):
        nid = n.get("id")
        if nid:
            node_ids.append(nid)
            node_meta[nid] = {
                "id": nid,
                "name": n.get("name", ""),
                "type": n.get("type", "")
            }
        
        fills = n.get("fills", [])
        if isinstance(fills, list):
            for p in fills:
                if isinstance(p, dict) and p.get("type") == "IMAGE":
                    ref = p.get("imageRef") or p.get("imageHash")
                    if ref:
                        image_refs.add(ref)
        
        strokes = n.get("strokes", [])
        if isinstance(strokes, list):
            for p in strokes:
                if isinstance(p, dict) and p.get("type") == "IMAGE":
                    ref = p.get("imageRef") or p.get("imageHash")
                    if ref:
                        image_refs.add(ref)
        
        for child in n.get("children", []) or []:
            visit(child)

    nodes = nodes_payload.get("nodes", {})
    for _, node in nodes.items():
        doc = node.get("document")
        if doc:
            visit(doc)
    
    log(f"Collected {len(image_refs)} image refs, {len(node_ids)} node IDs", option, "SUCCESS")
    return image_refs, node_ids, node_meta

def resolve_image_urls(file_key: str, token: str, image_refs: Set[str], node_ids_list: List[str], option: str) -> Tuple[Dict[str, str], Dict[str, Optional[str]]]:
    """Resolve image fill URLs and node render URLs from Figma API"""
    log("Resolving image URLs...", option, "INFO")
    headers = build_headers(token)
    
    fills_url = f"https://api.figma.com/v1/files/{file_key}/images"
    try:
        fills_resp = requests.get(fills_url, headers=headers, timeout=60)
        fills_resp.raise_for_status()
        fills_payload = fills_resp.json()
        full_map = fills_payload.get("images", {})
        filtered_fills = {k: v for k, v in full_map.items() if k in image_refs}
        log(f"Resolved {len(filtered_fills)} image fills", option, "SUCCESS")
    except Exception as e:
        log(f"Could not fetch image fills: {e}", option, "WARNING")
        filtered_fills = {}
    
    base_render_url = f"https://api.figma.com/v1/images/{file_key}"
    renders_map: Dict[str, Optional[str]] = {}
    
    batch_count = 0
    for batch in chunked(node_ids_list, RENDER_BATCH_SIZE):
        try:
            params = {"ids": ",".join(batch), "format": RENDER_FORMAT}
            r = requests.get(base_render_url, headers=headers, params=params, timeout=60)
            if r.ok:
                images_map = r.json().get("images", {})
                for nid in batch:
                    renders_map[nid] = images_map.get(nid)
                batch_count += 1
            else:
                for nid in batch:
                    renders_map[nid] = None
        except Exception:
            for nid in batch:
                renders_map[nid] = None
    
    log(f"Resolved {len(renders_map)} node renders in {batch_count} batches", option, "SUCCESS")
    return filtered_fills, renders_map

def log(message: str, option: str, level: str = "INFO"):
    """Simple logger function (replace with your own if already defined)."""
    print(f"[{level}] {option}: {message}")

def build_icon_map(
    nodes_data: Dict[str, Any],
    filtered_fills: Dict[str, str],
    renders_map: Dict[str, Optional[str]],
    node_meta: Dict[str, Dict[str, str]],
    option: str
) -> Dict[str, str]:
    """Build mapping from node IDs to image URLs"""
    log("Building icon map...", option, "INFO")
    
    def map_node_to_first_image_ref(n: dict, out: Dict[str, str]):
        nid = n.get("id")
        fills = n.get("fills", [])
        if isinstance(fills, list):
            for p in fills:
                if isinstance(p, dict) and p.get("type") == "IMAGE":
                    ref = p.get("imageRef") or p.get("imageHash")
                    if ref:
                        out[nid] = ref
                        break
        for child in n.get("children", []) or []:
            map_node_to_first_image_ref(child, out)

    node_first_ref: Dict[str, str] = {}
    nodes = nodes_data.get("nodes", {})
    for _, node in nodes.items():
        doc = node.get("document")
        if doc:
            map_node_to_first_image_ref(doc, node_first_ref)
    
    node_to_url: Dict[str, str] = {}
    for nid, meta in node_meta.items():
        url = None
        ref = node_first_ref.get(nid)
        if ref and isinstance(filtered_fills, dict):
            url = filtered_fills.get(ref)
        if not url:
            url = renders_map.get(nid)
        if url:
            node_to_url[nid] = url
    
    log(f"Built icon map with {len(node_to_url)} entries", option, "SUCCESS")
    return node_to_url


def merge_urls_into_nodes(
    nodes_data: Dict[str, Any],
    node_to_url: Dict[str, str],
    option: str
) -> Dict[str, Any]:
    """Merge image URLs into node structure"""
    log("Merging URLs into node structure...", option, "INFO")
    
    merged_nodes = copy.deepcopy(nodes_data)
    
    def inject_image_urls(n: dict):
        nid = n.get("id")
        if nid and nid in node_to_url:
            n["image_url"] = node_to_url[nid]
        for child in n.get("children", []) or []:
            inject_image_urls(child)
    
    nodes = merged_nodes.get("nodes", {})
    for _, node in nodes.items():
        doc = node.get("document")
        if doc:
            inject_image_urls(doc)
    
    log("URLs merged into node structure", option, "SUCCESS")
    return merged_nodes

# ========= UI Component Extraction Functions =========
from typing import Dict, Any, List, Optional

def find_document_roots(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Find document root nodes in Figma data"""
    roots: List[Dict[str, Any]] = []
    nodes = data.get('nodes')
    
    # Case 1: Figma file has a "nodes" dictionary with "document" subtrees
    if isinstance(nodes, dict) and nodes:
        for v in nodes.values():
            if isinstance(v, dict) and isinstance(v.get('document'), dict):
                roots.append(v['document'])
        if roots:
            return roots
    
    # Case 2: Direct "document" key in top-level data
    if isinstance(data.get('document'), dict):
        roots.append(data['document'])
        return roots
    
    # Case 3: Fallback – find first node with children
    def find_first_with_children(obj: Any) -> Optional[Dict[str, Any]]:
        if isinstance(obj, dict):
            if 'children' in obj and isinstance(obj['children'], list):
                return obj
            for val in obj.values():
                found = find_first_with_children(val)
                if found:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = find_first_with_children(item)
                if found:
                    return found
        return None
    
    candidate = find_first_with_children(data)
    if candidate:
        roots.append(candidate)
    
    return roots


def extract_bounds(node: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Extract position and size from node"""
    box = node.get('absoluteBoundingBox')
    if isinstance(box, dict):
        need = ['x', 'y', 'width', 'height']
        if all(k in box for k in need):
            return {k: box[k] for k in need}
    return None

def extract_layout(node: Dict[str, Any]) -> Dict[str, Any]:
    """Extract layout properties from node"""
    layout: Dict[str, Any] = {}
    for k in [
        'layoutMode', 'constraints', 'paddingLeft', 'paddingRight', 
        'paddingTop', 'paddingBottom', 'itemSpacing', 'counterAxisAlignItems',
        'primaryAxisAlignItems', 'layoutGrow', 'layoutAlign',
        'layoutSizingHorizontal', 'layoutSizingVertical',
        'counterAxisSizingMode', 'primaryAxisSizingMode',
        'clipsContent', 'layoutWrap'
    ]:
        if k in node:
            layout[k] = node[k]
    
    if 'layoutGrids' in node and is_nonempty_list(node['layoutGrids']):
        layout['layoutGrids'] = node['layoutGrids']
    
    return layout

def extract_visuals(node: Dict[str, Any]) -> Dict[str, Any]:
    """Extract visual styling properties from node"""
    styling: Dict[str, Any] = {}
    
    fills = node.get('fills')
    if is_nonempty_list(fills):
        parsed = []
        for fill in fills:
            ftype = (fill or {}).get('type')
            entry: Dict[str, Any] = {}
            if ftype == 'SOLID' and 'color' in fill:
                entry['type'] = 'solid'
                entry['color'] = to_rgba(fill['color'])
                if 'opacity' in fill:
                    entry['opacity'] = fill['opacity']
            elif ftype:
                entry['type'] = ftype.lower()
                if 'imageRef' in fill:
                    entry['imageRef'] = fill['imageRef']
                if 'scaleMode' in fill:
                    entry['scaleMode'] = fill['scaleMode']
                if 'gradientStops' in fill:
                    entry['gradientStops'] = fill['gradientStops']
            if entry:
                parsed.append(entry)
        if parsed:
            styling['fills'] = parsed
    
    if 'backgroundColor' in node and isinstance(node['backgroundColor'], dict):
        styling['backgroundColor'] = to_rgba(node['backgroundColor'])
    
    strokes = node.get('strokes')
    if is_nonempty_list(strokes):
        borders = []
        for stroke in strokes:
            s: Dict[str, Any] = {}
            if stroke.get('type') == 'SOLID' and 'color' in stroke:
                s['color'] = to_rgba(stroke['color'])
            if 'opacity' in stroke:
                s['opacity'] = stroke['opacity']
            if 'strokeWeight' in node:
                s['width'] = node['strokeWeight']
            if 'strokeAlign' in node:
                s['align'] = node['strokeAlign']
            if s:
                borders.append(s)
        if borders:
            styling['borders'] = borders
    
    if isinstance(node.get('cornerRadius'), (int, float)) and node['cornerRadius'] > 0:
        styling['cornerRadius'] = node['cornerRadius']
    
    effects = node.get('effects')
    if is_nonempty_list(effects):
        parsed = []
        for eff in effects:
            et = eff.get('type')
            if et:
                e: Dict[str, Any] = {'type': et.lower()}
                off = eff.get('offset') or {}
                if isinstance(off, dict):
                    e['x'] = off.get('x', 0)
                    e['y'] = off.get('y', 0)
                if 'radius' in eff:
                    e['blur'] = eff['radius']
                if 'color' in eff:
                    e['color'] = to_rgba(eff['color'])
                parsed.append(e)
        if parsed:
            styling['effects'] = parsed
    
    return styling

def extract_text(node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract text properties from text nodes"""
    if (node.get('type') or '').upper() != 'TEXT':
        return None
    
    text: Dict[str, Any] = {'content': node.get('characters', '')}
    style = node.get('style') or {}
    
    if isinstance(style, dict):
        text['typography'] = {
            'fontFamily': style.get('fontFamily'),
            'fontSize': style.get('fontSize'),
            'fontWeight': style.get('fontWeight'),
            'lineHeight': style.get('lineHeightPx', style.get('lineHeight')),
            'letterSpacing': style.get('letterSpacing'),
            'textAlign': (style.get('textAlignHorizontal') or 'left').lower(),
            'textCase': (style.get('textCase') or 'none').lower(),
        }
    
    fills = node.get('fills')
    if is_nonempty_list(fills):
        for f in fills:
            if f.get('type') == 'SOLID' and 'color' in f:
                text['color'] = to_rgba(f['color'])
                break
    
    return text

def should_include(node: Dict[str, Any]) -> bool:
    """Determine if node should be included in extraction"""
    t = (node.get('type') or '').upper()
    name = (node.get('name') or '')
    has_visual = bool(
        node.get('fills') or 
        node.get('strokes') or 
        node.get('effects') or 
        node.get('image_url')
    )
    semantic = any(k in name.lower() for k in [
        'button', 'input', 'search', 'nav', 'menu', 'container',
        'card', 'panel', 'header', 'footer', 'badge', 'chip'
    ])
    vector_visible = (
        t in ['VECTOR', 'LINE', 'ELLIPSE', 'POLYGON', 'STAR', 'RECTANGLE'] and 
        (node.get('strokes') or node.get('fills'))
    )
    
    return any([
        t == 'TEXT',
        has_visual,
        vector_visible,
        isinstance(node.get('cornerRadius'), (int, float)) and node.get('cornerRadius', 0) > 0,
        bool(node.get('layoutMode')),
        t in ['FRAME', 'GROUP', 'COMPONENT', 'INSTANCE', 'SECTION'],
        semantic
    ])

def classify_bucket(comp: Dict[str, Any]) -> str:
    """Classify component into appropriate bucket for Angular"""
    t = (comp.get('type') or '').upper()
    name = (comp.get('name') or '').lower()
    
    if t == 'TEXT':
        return 'textElements'
    if 'button' in name:
        return 'buttons'
    if any(k in name for k in ['input', 'search', 'textfield', 'field']):
        return 'inputs'
    if any(k in name for k in ['nav', 'menu', 'sidebar', 'toolbar', 'header', 'footer', 'app bar', 'breadcrumb']):
        return 'navigation'
    if comp.get('imageUrl'):
        return 'images'
    if t in ['VECTOR', 'LINE', 'ELLIPSE', 'POLYGON', 'STAR', 'RECTANGLE']:
        return 'vectors'
    if t in ['FRAME', 'GROUP', 'COMPONENT', 'INSTANCE', 'SECTION'] or any(k in name for k in ['container', 'card', 'panel', 'section']):
        return 'containers'
    
    return 'other'

def extract_components(
    root: Dict[str, Any],
    parent_path: str = "",
    out: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """Recursively extract components from node tree"""
    if out is None:
        out = []
    
    path = f"{parent_path}/{root.get('name', 'Unnamed')}" if parent_path else (root.get('name') or 'Root')
    
    comp: Dict[str, Any] = {
        'id': root.get('id'),
        'name': root.get('name'),
        'type': root.get('type'),
        'path': path,
    }
    
    bounds = extract_bounds(root)
    if bounds:
        comp['position'] = bounds
    
    layout = extract_layout(root)
    if layout:
        comp['layout'] = layout
    
    styling = extract_visuals(root)
    if styling:
        comp['styling'] = styling
    
    if 'image_url' in root:
        comp['imageUrl'] = root['image_url']
    
    text = extract_text(root)
    if text:
        comp['text'] = text
    
    if should_include(root):
        out.append(comp)
    
    for child in root.get('children', []) or []:
        if isinstance(child, dict):
            extract_components(child, path, out)
    
    return out

def organize_for_angular(components: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Organize extracted components into Angular-friendly structure"""
    organized = {
        'metadata': {
            'totalComponents': len(components),
            'extractedAt': datetime.datetime.utcnow().isoformat() + 'Z',
            'version': 1
        },
        'textElements': [],
        'buttons': [],
        'inputs': [],
        'containers': [],
        'images': [],
        'navigation': [],
        'vectors': [],
        'other': []
    }
    
    for c in components:
        organized[classify_bucket(c)].append(c)
    
    return organized
from typing import Dict, Any, List

def extract_ui_components(
    merged_data: Dict[str, Any],
    option: str
) -> Dict[str, Any]:
    """Extract and organize UI components from merged Figma data"""
    log("Extracting UI components...", option, "INFO")
    
    roots = find_document_roots(merged_data)
    if not roots:
        raise RuntimeError("No document roots found in JSON")
    
    all_components: List[Dict[str, Any]] = []
    for root in roots:
        if isinstance(root, dict):
            extract_components(root, "", all_components)
    
    organized = organize_for_angular(all_components)
    log(f"Extracted {len(all_components)} components", option, "SUCCESS")
    
    return organized


def process_figma_extraction(
    file_key: str,
    node_ids: str,
    token: str,
    option: str
) -> Dict[str, Any]:
    """Complete Figma extraction pipeline"""
    try:
        nodes_data = fetch_figma_nodes(file_key, node_ids, token, option)
        image_refs, node_ids_list, node_meta = walk_nodes_collect_images_and_ids(nodes_data, option)
        filtered_fills, renders_map = resolve_image_urls(file_key, token, image_refs, node_ids_list, option)
        node_to_url = build_icon_map(nodes_data, filtered_fills, renders_map, node_meta, option)
        merged_data = merge_urls_into_nodes(nodes_data, node_to_url, option)
        final_output = extract_ui_components(merged_data, option)
        
        meta = final_output.get("metadata", {})
        total_components = meta.get("totalComponents", len(final_output.get("components", [])))
        log(f"✓ Extraction completed: {total_components} components", option, "SUCCESS")
        
        return final_output

    except Exception as e:
        log(f"Extraction pipeline error: {str(e)}", option, "ERROR")
        raise


# ========= JSON Helper Functions =========
def save_json(data_obj: dict, path: Path, option: str) -> bool:
    """Save JSON to file"""
    try:
        with open(str(path), "w", encoding="utf-8") as f:
            json.dump(data_obj, f, ensure_ascii=False, indent=2)
        
        file_size = path.stat().st_size
        log(f"✓ JSON saved: {path.name} ({file_size:,} bytes)", option, "SUCCESS")
        return True
    except Exception as e:
        log(f"Failed to save {path.name}: {str(e)}", option, "ERROR")
        return False

def split_json_chunks(data_obj: dict, option: str) -> Tuple[dict, dict]:
    """Split JSON into two balanced chunks"""
    log("Splitting JSON into chunks...", option, "INFO")
    
    cats = ['textElements', 'buttons', 'inputs', 'containers', 
            'images', 'navigation', 'vectors', 'other']
    
    all_items: List[Tuple[str, Any]] = []
    for cat in cats:
        if isinstance(data_obj.get(cat), list):
            items = data_obj.get(cat, [])
            all_items.extend((cat, item) for item in items)
    
    split_idx = len(all_items) // 2
    
    chunk1_data = {cat: [] for cat in cats}
    chunk2_data = {cat: [] for cat in cats}
    
    for idx, (cat, obj) in enumerate(all_items):
        if idx < split_idx:
            chunk1_data[cat].append(obj)
        else:
            chunk2_data[cat].append(obj)
    
    meta = data_obj.get('metadata', {})
    chunk1_data['metadata'] = {**meta, 'chunkIndex': 0, 'totalChunks': 2}
    chunk2_data['metadata'] = {**meta, 'chunkIndex': 1, 'totalChunks': 2}
    
    chunk1_size = sum(len(chunk1_data[cat]) for cat in cats)
    chunk2_size = sum(len(chunk2_data[cat]) for cat in cats)
    log(f"✓ Chunk 1: {chunk1_size} items, Chunk 2: {chunk2_size} items", option, "SUCCESS")
    
    return chunk1_data, chunk2_data

def display_component_summary( data: dict) -> str:
    """Generate component summary text"""
    categories = ['textElements', 'buttons', 'inputs', 'containers',
                  'images', 'navigation', 'vectors', 'other']
    
    total = data.get('metadata', {}).get('totalComponents', 0)
    
    summary = f"✓ Total components: {total}\n" + "="*60 + "\n\nComponent Summary:\n"
    
    for cat in categories:
        items = data.get(cat, [])
        if isinstance(items, list) and len(items) > 0:
            summary += f"  - {cat}: {len(items)}\n"
    
    return summary

# ========= CrewAI Integration Functions =========
def init_llm(option: str) -> Optional["LLM"]:
    """Initialize LLM for CrewAI agents"""
    if not CREW_AVAILABLE:
        log("CrewAI library not available", option, "ERROR")
        return None
    
    if not GEMINI_API_KEY:
        log("Gemini API key not configured", option, "ERROR")
        return None
    
    try:
        log(f"Initializing LLM: {LLM_MODEL_ID}", option, "INFO")
        llm = LLM(
            model=LLM_MODEL_ID,
            api_key=GEMINI_API_KEY,
            max_tokens=MAX_OUTPUT_TOKENS,
            timeout=DEFAULT_TIMEOUT
        )
        log("✓ LLM initialized successfully", option, "SUCCESS")
        return llm
    except Exception as e:
        log(f"LLM initialization failed: {str(e)}", option, "ERROR")
        return None

def create_angular_converter_agent(llm) -> Agent:
    """Create agent for Angular code conversion"""
    return Agent(
        role="Angular Developer",
        goal="Convert Figma JSON metadata to Angular 19 standalone components with pixel-perfect responsive design",
        backstory="Senior Angular developer specializing in modern component architecture and UI conversion from design systems",
        llm=llm,
        verbose=False,
        allow_delegation=False
    )

def create_angular_merge_agent(llm) -> Agent:
    """Create agent for merging Angular chunks"""
    return Agent(
        role="Angular Architect",
        goal="Merge multiple Angular code chunks into a single cohesive standalone Angular 19 application with proper routing and structure",
        backstory="Lead architect specializing in Angular application structure, routing, module integration, and production deployments",
        llm=llm,
        verbose=False,
        allow_delegation=False
    )

def run_angular_conversion(agent: Agent, json_path: Path, label: str, option: str) -> str:
    """Run Angular conversion task on JSON file"""
    try:
        log(f"Starting Angular conversion for {label}...", option, "INFO")
        
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        
        json_str = json.dumps(json_data, indent=2)
        
        if len(json_str) > 150000:
            json_str = json_str[:150000] + "\n... [truncated for processing]"
            log(f"JSON truncated to 150K chars for {label}", option, "WARNING")
        
        prompt = f"""Convert the following Figma JSON metadata ({label}) into Angular 19 standalone components.

Requirements:
- Create standalone components (no modules)
- Include TypeScript (.ts), HTML templates (.html), and CSS styles (.css)
- Use modern Angular 19 features and syntax
- Ensure responsive design with proper breakpoints
- Follow Angular style guide and best practices
- Include proper type definitions and interfaces
- Use meaningful component and file names

Output the complete Angular code in markdown format with proper code blocks for each file.

JSON Meta
{json_str}
"""
        
        task = Task(
            description=prompt,
            expected_output="Complete Angular 19 component code in markdown format with TypeScript, HTML, and CSS files",
            agent=agent
        )
        
        log(f"Executing CrewAI task for {label}...", option, "INFO")
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False
        )
        
        result = crew.kickoff()
        
        log(f"✓ Angular conversion completed for {label}", option, "SUCCESS")
        return str(result)
        
    except Exception as e:
        error_msg = f"Angular conversion failed for {label}: {str(e)}"
        log(error_msg, option, "ERROR")
        return f"ERROR: {error_msg}"

def run_merge_conversion(agent: Agent, chunk1_path: Path, chunk2_path: Path, option: str) -> str:
    """Merge two Angular code chunks into single application"""
    try:
        log("Starting Angular merge process...", option, "INFO")
        
        with open(chunk1_path, "r", encoding="utf-8") as f:
            code1 = f.read()
        
        with open(chunk2_path, "r", encoding="utf-8") as f:
            code2 = f.read()
        
        code1_limited = code1[:80000] if len(code1) > 80000 else code1
        code2_limited = code2[:80000] if len(code2) > 80000 else code2
        
        if len(code1) > 80000 or len(code2) > 80000:
            log("Angular chunks truncated for merge processing", option, "WARNING")
        
        prompt = f"""Merge the following two Angular code chunks into a single cohesive Angular 19 standalone application.

Requirements:
- Combine all components without duplication
- Create proper application structure with main.ts, app.config.ts, and app.component.ts
- Set up routing using provideRouter if navigation components exist
- Resolve any naming conflicts intelligently
- Ensure all imports are correct and paths are valid
- Create a complete, deployable Angular 19 application
- Follow Angular best practices and style guide
- Use standalone components throughout (no NgModule)

Output the complete merged Angular application code in markdown format with all necessary files.

CHUNK 1 CODE:
{code1_limited}

CHUNK 2 CODE:
{code2_limited}
"""
        
        task = Task(
            description=prompt,
            expected_output="Complete merged Angular 19 standalone application in markdown format with all files",
            agent=agent
        )
        
        log("Executing CrewAI merge task...", option, "INFO")
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False
        )
        
        result = crew.kickoff()
        
        log("✓ Angular merge completed successfully", option, "SUCCESS")
        return str(result)
        
    except Exception as e:
        error_msg = f"Angular merge failed: {str(e)}"
        log(error_msg, option, "ERROR")
        return f"ERROR: {error_msg}"

# ========= Main Streamlit Application =========
def main():
    # Custom CSS for attractive header with gradient background
    st.markdown("""
        <style>
            /* Main header container with gradient */
            .main-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 75%, #4facfe 100%);
                padding: 2.5rem 2rem;
                border-radius: 15px;
                margin-bottom: 2rem;
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
                text-align: center;
                animation: gradient-shift 8s ease infinite;
                background-size: 300% 300%;
            }
            
            @keyframes gradient-shift {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }
            
            .main-header h1 {
                color: white;
                font-size: 2.8rem;
                font-weight: 800;
                margin: 0;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
                letter-spacing: -0.5px;
            }
            
            .main-header p {
                color: rgba(255, 255, 255, 0.95);
                font-size: 1.1rem;
                margin-top: 0.8rem;
                font-weight: 500;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
            }
            
            /* Feature badges styling */
            .stMarkdown [data-testid="stMarkdownContainer"] {
                font-size: 0.95rem;
            }
            
            /* Tab styling */
            div[data-baseweb="tab-list"] button {
                width: 100% !important;
                justify-content: center !important;
                font-size: 15px !important;
                font-weight: 600 !important;
                padding: 12px !important;
                border-radius: 10px !important;
            }
            
            div[data-baseweb="tab-list"] button[aria-selected="true"] {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                color: white !important;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Attractive header with gradient background
    st.markdown("""
        <div class="main-header">
            <h1> Figma Wireframe to Angular Converter</h1>
            <p>Professional Multi-Agent Workflow – Convert Figma wireframes into production-ready Angular 19 code</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Information
    with st.sidebar:
        st.markdown("## 🗂️ Application Information")
        st.markdown("#### Overview of available LLM models and settings")
        st.markdown("---")
        
        st.markdown("### 🤖 AI Model Configuration")
        st.info(f"""
        **Model**: {LLM_MODEL_ID}  
        **Max Tokens**: {MAX_OUTPUT_TOKENS:,}  
        **Timeout**: {DEFAULT_TIMEOUT}s  
        **Provider**: Google Gemini  
        **CrewAI**: {'✅ Available' if CREW_AVAILABLE else '❌ Not Available'}
        """)
        
        st.markdown("---")
        st.markdown("### 🏗️ Workflow Architecture")
        st.markdown("**Multi-Agent System**")
        st.markdown("""
        - Extraction Agent
        - Conversion Agent
        - Merge Agent (Large files)
        - Validation Agent
        """)
        
        st.markdown("---")
        st.markdown("### ⚙️ System Requirements")
        st.markdown("""
        **Angular**: 19.x  
        **Node.js**: 18.x or higher  
        **TypeScript**: 5.x  
        **CSS**: Grid & Flexbox  
        **API**: Figma REST API
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ How to Use")
        st.markdown("""
        **Option 1 - Small File:**
        1. Enter Figma credentials
        2. Extract metadata
        3. Generate Angular code
        
        **Option 2 - Large File:**
        1. Enter Figma credentials
        2. Extract & split metadata
        3. Process Chunk 1
        4. Process Chunk 2
        5. Generate final app
        """)
    
    # Feature badges
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("🤖 **Multi-Agent Architecture**")
    with col2:
        st.markdown("⚡ **Real-Time Processing**")
    with col3:
        st.markdown("📊 **Component Analytics**")
    with col4:
        st.markdown("🔒 **Secure Token Handling**")
    
    col5, col6, col7 = st.columns(3)
    with col5:
        st.markdown("📦 **Chunk Management**")
    with col6:
        st.markdown("🎯 **Pixel-Perfect Output**")
    with col7:
        st.markdown("📥 **Multi-Format Export**")
        
    
    st.markdown("---")
    
    # Workflow descriptions
    st.markdown("## 📦 Small File Processing Workflow")
    st.info("""
    **💡 Best For:** Simple wireframes, individual screens, component libraries, prototypes with limited complexity  
    **⚡ Processing:** Single-agent streamlined workflow for optimal speed  
    **⏱️ Avg Time:** 5-10 minutes 
    """)
    
    st.markdown("")
    
    st.markdown("## 🚀 Large File Processing Workflow")
    st.info("""
    **💡 Best For:** Complex applications, multi-screen flows, design systems, enterprise-level projects  
    **⚡ Processing:** Advanced multi-agent workflow with intelligent chunking  
    **⏱️ Avg Time:** 10-15 minutes 
    """)
    
    st.markdown("---")
    
    # Main Content - Tabs with spacing
    tab_small, tab_large = st.tabs([
        "📄 Standard Figma File",
        "🚀 Enterprise Figma File"
    ])

    
    # ========= OPTION 1: SMALL FILE =========
    with tab_small:
        st.header("Option 1: Small Figma File Processing")
        st.markdown("Process Figma files with less than 1000 components using a streamlined single-agent workflow.")
        
        st.markdown("---")
        st.subheader("Step 1: Figma Credentials")
        
        col1, col2 = st.columns(2)
        with col1:
            small_file_key = st.text_input(
                "Figma File Key",
                key="small_file_key",
                help="The file key from your Figma URL",
                placeholder="Enter Figma file key..."
            )
        
        with col2:
            small_node_ids = st.text_input(
                "Node IDs (comma-separated)",
                key="small_node_ids",
                help="Comma-separated list of node IDs to extract",
                placeholder="0:1, 10:23, 5:1"
            )
        
        small_token = st.text_input(
            "Personal Access Token",
            key="small_token",
            type="password",
            help="Your Figma personal access token",
            placeholder="Enter your Figma access token..."
        )
        
        st.markdown("---")
        st.subheader("Step 2: Extract Metadata")
        
        btn_extract_small = st.button(
            "🔄 Extract & Save Figma Metadata",
            key="btn_extract_small",
            use_container_width=True,
            type="primary"
        )
        
        if btn_extract_small:
            valid, errors = validate_figma_inputs(small_file_key, small_node_ids, small_token)
            
            if not valid:
                st.error("❌ Validation Failed")
                for error in errors:
                    st.error(f"• {error}")
            else:
                # Reset state
                clear_logs("small")
                st.session_state.small_metadata = None
                st.session_state.angular_small = None
                st.session_state.metadata_extracted_small = False
                
                prepare_directories_small("small")
                
                with st.spinner("🔄 Extracting Figma metadata..."):
                    try:
                        extracted_data = process_figma_extraction(
                            small_file_key.strip(),
                            small_node_ids.strip(),
                            small_token.strip(),
                            "small"
                        )
                        
                        json_path = SMALL_DIR / "small_metadata.json"
                        
                        if save_json(extracted_data, json_path, "small"):
                            st.session_state.small_metadata = extracted_data
                            st.session_state.metadata_extracted_small = True
                            
                            st.success("✅ Metadata extracted and saved successfully!")
                            
                            # Display summary
                            summary_text = display_component_summary(extracted_data)
                            st.code(summary_text, language="text")
                            
                            # Download button
                            st.download_button(
                                label="📥 Download small_metadata.json",
                                data=json.dumps(extracted_data, indent=2),
                                file_name="small_metadata.json",
                                mime="application/json",
                                use_container_width=True
                            )
                        else:
                            st.error("❌ Failed to save metadata file")
                    
                    except Exception as e:
                        st.error(f"❌ Extraction failed: {str(e)}")
        
        # Show logs
        if len(st.session_state.logs_small) > 0:
            with st.expander("📋 View Execution Logs (Small File)", expanded=False):
                render_logs("small")
        
        st.markdown("---")
        st.subheader("Step 3: Generate Angular Code")
        
        btn_generate_small = st.button(
            "⚡ Generate Angular Components",
            key="btn_generate_small",
            disabled=not st.session_state.metadata_extracted_small,
            use_container_width=True,
            type="primary"
        )
        
        if btn_generate_small:
            with st.spinner("⚡ Converting to Angular code..."):
                try:
                    llm = init_llm("small")
                    
                    if not llm:
                        st.error("❌ Failed to initialize LLM")
                    else:
                        agent = create_angular_converter_agent(llm)
                        json_path = SMALL_DIR / "small_metadata.json"
                        
                        result = run_angular_conversion(agent, json_path, "small_metadata", "small")
                        
                        if not result.startswith("ERROR"):
                            st.session_state.angular_small = result
                            
                            output_path = SMALL_DIR / "angular.md"
                            with open(output_path, "w", encoding="utf-8") as f:
                                f.write(result)
                            
                            st.success("✅ Angular code generated successfully!")
                        else:
                            st.error(f"❌ Conversion failed: {result}")
                
                except Exception as e:
                    st.error(f"❌ Generation failed: {str(e)}")
        
        # Show logs after generation
        if len(st.session_state.logs_small) > 0:
            with st.expander("📋 View Execution Logs (Small File)", expanded=False):
                render_logs("small")
        
        # Display Angular code output
        if st.session_state.angular_small:
            st.markdown("---")
            st.subheader("📄 Angular Code Output")
            
            with st.expander("👀 Preview Angular Code", expanded=False):
                preview_text = st.session_state.angular_small[:5000]
                if len(st.session_state.angular_small) > 5000:
                    preview_text += "\n... [truncated for display]"
                st.code(preview_text, language="markdown")
            
            st.markdown("### 📥 Download Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    label="📄 Download as Markdown (.md)",
                    data=st.session_state.angular_small,
                    file_name="angular_small.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            
            with col2:
                st.download_button(
                    label="📝 Download as Text (.txt)",
                    data=st.session_state.angular_small,
                    file_name="angular_small.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col3:
                pdf_data = build_pdf("Angular Code - Small File", st.session_state.angular_small)
                if pdf_data:
                    st.download_button(
                        label="📕 Download as PDF (.pdf)",
                        data=pdf_data,
                        file_name="angular_small.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
    
    # ========= OPTION 2: LARGE FILE =========
    with tab_large:
        st.header("Option 2: Large Figma File Processing")
        st.markdown("Process Figma files with 1000+ components using advanced chunking and multi-agent workflow.")
        
        st.markdown("---")
        st.subheader("Step 1: Figma Credentials")
        
        col1, col2 = st.columns(2)
        with col1:
            large_file_key = st.text_input(
                "Figma File Key",
                key="large_file_key",
                help="The file key from your Figma URL",
                placeholder="Enter Figma file key..."
            )
        
        with col2:
            large_node_ids = st.text_input(
                "Node IDs (comma-separated)",
                key="large_node_ids",
                help="Comma-separated list of node IDs to extract",
                placeholder="0:1, 10:23, 5:1"
            )
        
        large_token = st.text_input(
            "Personal Access Token",
            key="large_token",
            type="password",
            help="Your Figma personal access token",
            placeholder="Enter your Figma access token..."
        )
        
        st.markdown("---")
        st.subheader("Step 2: Extract & Split Metadata")
        
        btn_extract_large = st.button(
            "🔄 Extract & Split Figma Metadata",
            key="btn_extract_large",
            use_container_width=True,
            type="primary"
        )
        
        if btn_extract_large:
            valid, errors = validate_figma_inputs(large_file_key, large_node_ids, large_token)
            
            if not valid:
                st.error("❌ Validation Failed")
                for error in errors:
                    st.error(f"• {error}")
            else:
                # Reset state
                clear_logs("large")
                st.session_state.large_metadata = None
                st.session_state.large_chunk1_json = None
                st.session_state.large_chunk2_json = None
                st.session_state.angular_chunk1 = None
                st.session_state.angular_chunk2 = None
                st.session_state.angular_merged_app = None
                st.session_state.chunk1_done = False
                st.session_state.chunk2_done = False
                st.session_state.metadata_extracted_large = False
                
                prepare_directories_large("large")
                
                with st.spinner("🔄 Extracting and splitting Figma metadata..."):
                    try:
                        extracted_data = process_figma_extraction(
                            large_file_key.strip(),
                            large_node_ids.strip(),
                            large_token.strip(),
                            "large"
                        )
                        
                        metadata_path = LARGE_DIR / "large_metadata.json"
                        
                        if save_json(extracted_data, metadata_path, "large"):
                            st.session_state.large_metadata = extracted_data
                            
                            chunk1_data, chunk2_data = split_json_chunks(extracted_data, "large")
                            
                            st.session_state.large_chunk1_json = chunk1_data
                            st.session_state.large_chunk2_json = chunk2_data
                            
                            chunk1_path = CHUNK1_DIR / "chunk1.json"
                            chunk2_path = CHUNK2_DIR / "chunk2.json"
                            
                            ok1 = save_json(chunk1_data, chunk1_path, "large")
                            ok2 = save_json(chunk2_data, chunk2_path, "large")
                            
                            if ok1 and ok2:
                                st.session_state.metadata_extracted_large = True
                                
                                st.success("✅ Metadata extracted and split successfully!")
                                
                                # Display summary
                                summary_text = display_component_summary(extracted_data)
                                st.code(summary_text, language="text")
                                
                                # Download buttons
                                st.markdown("### 📥 Download Metadata Files")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.download_button(
                                        label="📥 large_metadata.json",
                                        data=json.dumps(extracted_data, indent=2),
                                        file_name="large_metadata.json",
                                        mime="application/json",
                                        use_container_width=True
                                    )
                                
                                with col2:
                                    st.download_button(
                                        label="📥 chunk1.json",
                                        data=json.dumps(chunk1_data, indent=2),
                                        file_name="chunk1.json",
                                        mime="application/json",
                                        use_container_width=True
                                    )
                                
                                with col3:
                                    st.download_button(
                                        label="📥 chunk2.json",
                                        data=json.dumps(chunk2_data, indent=2),
                                        file_name="chunk2.json",
                                        mime="application/json",
                                        use_container_width=True
                                    )
                            else:
                                st.error("❌ Failed to save chunk files")
                        else:
                            st.error("❌ Failed to save metadata file")
                    
                    except Exception as e:
                        st.error(f"❌ Extraction failed: {str(e)}")
        
        # Show logs
        if len(st.session_state.logs_large) > 0:
            with st.expander("📋 View Execution Logs (Large File)", expanded=False):
                render_logs("large")
        
        st.markdown("---")
        st.subheader("Step 3: Process Chunks")
        
        col1, col2 = st.columns(2)
        
        # -------- Buttons for Chunk 1 & 2 --------
        with col1:
            btn_process_chunk1 = st.button(
                "⚡ Process Chunk 1",
                key="btn_process_chunk1",
                disabled=not st.session_state.metadata_extracted_large,
                use_container_width=True,
                type="primary",
            )
        
        with col2:
            btn_process_chunk2 = st.button(
                "⚡ Process Chunk 2",
                key="btn_process_chunk2",
                disabled=not st.session_state.chunk1_done,
                use_container_width=True,
                type="primary",
            )
        
        # -------- Chunk 1 Processing --------
        if btn_process_chunk1:
            with st.spinner("⚡ Converting Chunk 1 to Angular code..."):
                try:
                    llm = init_llm("large")
                    
                    if not llm:
                        st.error("❌ Failed to initialize LLM")
                    else:
                        agent = create_angular_converter_agent(llm)
                        chunk1_path = CHUNK1_DIR / "chunk1.json"
                        
                        result = run_angular_conversion(agent, chunk1_path, "chunk1", "large")
                        
                        if not result.startswith("ERROR"):
                            st.session_state.angular_chunk1 = result
                            
                            output_path = CHUNK1_DIR / "chunk1_angular.md"
                            with open(output_path, "w", encoding="utf-8") as f:
                                f.write(result)
                            
                            st.session_state.chunk1_done = True
                            st.success("✅ Chunk 1 processed successfully! You can now process Chunk 2.")
                            st.rerun()  # 🔥 Enables Chunk 2 immediately
                        else:
                            st.error(f"❌ Chunk 1 conversion failed: {result}")
                
                except Exception as e:
                    st.error(f"❌ Chunk 1 processing failed: {str(e)}")
        
        # -------- Chunk 2 Processing --------
        if btn_process_chunk2:
            with st.spinner("⚡ Converting Chunk 2 to Angular code..."):
                try:
                    llm = init_llm("large")
                    
                    if not llm:
                        st.error("❌ Failed to initialize LLM")
                    else:
                        agent = create_angular_converter_agent(llm)
                        chunk2_path = CHUNK2_DIR / "chunk2.json"
                        
                        result = run_angular_conversion(agent, chunk2_path, "chunk2", "large")
                        
                        if not result.startswith("ERROR"):
                            st.session_state.angular_chunk2 = result
                            
                            output_path = CHUNK2_DIR / "chunk2_angular.md"
                            with open(output_path, "w", encoding="utf-8") as f:
                                f.write(result)
                            
                            st.session_state.chunk2_done = True
                            st.success("✅ Chunk 2 processed successfully! You can now generate the final Angular app.")
                            st.rerun()  # 🔥 Enables Generate Final button immediately
                        else:
                            st.error(f"❌ Chunk 2 conversion failed: {result}")
                
                except Exception as e:
                    st.error(f"❌ Chunk 2 processing failed: {str(e)}")
        
        # -------- Logs Section --------
        if len(st.session_state.logs_large) > 0:
            with st.expander("📋 View Execution Logs (Large File)", expanded=False):
                render_logs("large")
        
        # -------- Final Angular App Generation --------
        st.markdown("---")
        st.subheader("Step 4: Generate Final Angular Application")
        
        btn_generate_final = st.button(
            "🚀 Generate Final Angular App",
            key="btn_generate_final",
            disabled=not (st.session_state.chunk1_done and st.session_state.chunk2_done),
            use_container_width=True,
            type="primary",
        )
        
        if btn_generate_final:
            with st.spinner("🚀 Generating final Angular application..."):
                try:
                    llm = init_llm("large")
                    
                    if not llm:
                        st.error("❌ Failed to initialize LLM")
                    else:
                        agent = create_angular_merge_agent(llm)
                        
                        chunk1_angular_path = CHUNK1_DIR / "chunk1_angular.md"
                        chunk2_angular_path = CHUNK2_DIR / "chunk2_angular.md"
                        
                        result = run_merge_conversion(agent, chunk1_angular_path, chunk2_angular_path, "large")
                        
                        if not result.startswith("ERROR"):
                            st.session_state.angular_merged_app = result
                            
                            output_path = LARGE_DIR / "merged_angular.md"
                            with open(output_path, "w", encoding="utf-8") as f:
                                f.write(result)
                            
                            st.success("✅ Final Angular application generated successfully!")
                        else:
                            st.error(f"❌ Merge failed: {result}")
                
                except Exception as e:
                    st.error(f"❌ Final generation failed: {str(e)}")
        
        # -------- Show Logs & Output --------
        if len(st.session_state.logs_large) > 0:
            with st.expander("📋 View Execution Logs (Large File)", expanded=False):
                render_logs("large")
        
        if st.session_state.angular_merged_app:
            st.markdown("---")
            st.subheader("📄 Final Angular Application")
            
            with st.expander("👀 Preview Angular Code", expanded=False):
                preview_text = st.session_state.angular_merged_app[:5000]
                if len(st.session_state.angular_merged_app) > 5000:
                    preview_text += "\n... [truncated for display]"
                st.code(preview_text, language="markdown")
            
            st.markdown("### 📥 Download Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    label="📄 Download as Markdown (.md)",
                    data=st.session_state.angular_merged_app,
                    file_name="angular_merged.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            
            with col2:
                st.download_button(
                    label="📝 Download as Text (.txt)",
                    data=st.session_state.angular_merged_app,
                    file_name="angular_merged.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            
            with col3:
                pdf_data = build_pdf("Angular Code - Large File (Merged)", st.session_state.angular_merged_app)
                if pdf_data:
                    st.download_button(
                        label="📕 Download as PDF (.pdf)",
                        data=pdf_data,
                        file_name="angular_merged.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )


# ========= Entry =========
if __name__ == "__main__":
    main()

