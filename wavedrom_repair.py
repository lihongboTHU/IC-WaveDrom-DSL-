import json
import re


DEFAULT_WAVEDROM_PROMPT = (
    "Return only one valid WaveDrom JSON object. "
    "Do not output extra signals. Do not include time-axis labels."
)

VALID_WAVE_CHARS = set("01xzpnPNhlud=2345.|")
WAVE_LIKE_RE = re.compile(r"^[01xzpnPNhlud=2345.|]+$")


def normalize_json_text(text):
    """Strip chat/fence noise and keep the likely JSON region."""
    if text is None:
        return ""

    text = str(text).strip()
    text = text.replace("```json", "").replace("```JSON", "").replace("```", "")

    first = text.find("{")
    if first > 0:
        text = text[first:]
    return text.strip()


def _patch_common_syntax_errors(text):
    # Model sometimes emits {"name":"CLK":"p...."} instead of adding a wave key.
    text = re.sub(
        r'("name"\s*:\s*"[^"]+")\s*:\s*"([01xzpnPNhlud=2345.|]+)"',
        r'\1,"wave":"\2"',
        text,
    )
    # Drop dangling commas before array/object closures.
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def _try_load(text):
    try:
        return json.loads(text)
    except Exception:
        return None


def _balanced_prefix(text):
    """Return the longest prefix that ends at a balanced top-level object."""
    start = text.find("{")
    if start < 0:
        return ""

    depth = 0
    in_string = False
    escape = False
    last_balanced = -1

    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                last_balanced = idx + 1
            elif depth < 0:
                break

    if last_balanced > 0:
        return text[start:last_balanced]
    return text[start:]


def _close_truncated_json(text):
    """Best-effort close for outputs truncated after a valid prefix."""
    text = text.strip()
    if not text:
        return text

    in_string = False
    escape = False
    stack = []

    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()

    if in_string:
        text += '"'
    while stack:
        closer = stack.pop()
        if text.rstrip().endswith(","):
            text = text.rstrip()[:-1]
        text += closer
    return text


def _extract_signal_objects(text):
    marker = re.search(r'"signal"\s*:\s*\[', text)
    if not marker:
        return []

    start = text.find("[", marker.start())
    decoder = json.JSONDecoder()
    pos = start + 1
    signals = []

    while pos < len(text):
        while pos < len(text) and text[pos] in " \r\n\t,":
            pos += 1
        if pos >= len(text) or text[pos] == "]":
            break
        if text[pos] != "{":
            pos += 1
            continue

        obj = None
        try:
            obj, end = decoder.raw_decode(text[pos:])
            pos += end
        except Exception:
            fragment = _balanced_prefix(text[pos:])
            if fragment and fragment != text[pos:]:
                obj = _try_load(_patch_common_syntax_errors(fragment))
                pos += len(fragment)
            else:
                obj = _try_load(_close_truncated_json(_patch_common_syntax_errors(text[pos:])))
                pos = len(text)

        if isinstance(obj, dict):
            signals.append(obj)
        else:
            pos += 1

    return signals


def _looks_like_generated_junk(sig):
    name = str(sig.get("name", "")).strip()
    wave = str(sig.get("wave", "")).strip()
    return not wave and len(name) >= 4 and WAVE_LIKE_RE.fullmatch(name)


def sanitize_wavedrom_obj(obj, max_signals=32):
    if not isinstance(obj, dict):
        obj = {"signal": []}

    raw_signals = obj.get("signal", [])
    if not isinstance(raw_signals, list):
        raw_signals = []

    cleaned = []
    seen = set()

    for item in raw_signals:
        if isinstance(item, list):
            for sub in item:
                if isinstance(sub, dict):
                    raw_signals.append(sub)
            continue
        if not isinstance(item, dict) or _looks_like_generated_junk(item):
            continue

        name = str(item.get("name", "")).strip()
        wave = "".join(ch for ch in str(item.get("wave", "")) if ch in VALID_WAVE_CHARS)
        if not name and not wave:
            continue

        sig = {"name": name}
        if wave:
            sig["wave"] = wave

        data = item.get("data", [])
        if isinstance(data, str):
            data = [data]
        if isinstance(data, list):
            data = [str(x).strip() for x in data if str(x).strip()]
            equal_count = wave.count("=")
            if equal_count:
                data = data[:equal_count]
            else:
                data = []
            if data:
                sig["data"] = data

        phase = item.get("phase")
        if isinstance(phase, (int, float)):
            sig["phase"] = phase

        key = json.dumps(sig, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(sig)
        if len(cleaned) >= max_signals:
            break

    return {"signal": cleaned}


def repair_wavedrom_json(text):
    """Return (json_text, obj, repair_info). obj is None only if no signal object can be recovered."""
    info = {"used_repair": False, "method": "direct"}
    normalized = normalize_json_text(text)
    clean = _patch_common_syntax_errors(normalized)
    prepatched = clean != normalized

    candidates = []
    balanced = _balanced_prefix(clean)
    if balanced:
        candidates.append(balanced)
    closed = _close_truncated_json(balanced or clean)
    if closed and closed not in candidates:
        candidates.append(closed)

    for candidate in candidates:
        obj = _try_load(candidate)
        if isinstance(obj, dict):
            sanitized = sanitize_wavedrom_obj(obj)
            repaired_text = json.dumps(sanitized, ensure_ascii=False, separators=(",", ":"))
            info["used_repair"] = prepatched or repaired_text != candidate
            info["method"] = "direct_or_sanitized"
            return repaired_text, sanitized, info

    signals = _extract_signal_objects(clean)
    if signals:
        sanitized = sanitize_wavedrom_obj({"signal": signals})
        info["used_repair"] = True
        info["method"] = "partial_signal_extract"
        return json.dumps(sanitized, ensure_ascii=False, separators=(",", ":")), sanitized, info

    return clean, None, {"used_repair": True, "method": "failed"}
