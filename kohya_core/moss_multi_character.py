"""MOSS multi-character validation layouts and prompt-file handling (CPU only)."""
from __future__ import annotations

import json

CLASS_WORDS = ("person", "woman", "man", "girl", "boy")


def options(settings):
    value = settings.get("moss", {})
    if not isinstance(value, dict):
        raise ValueError("MOSS 多角色选项必须是对象。")
    result = {"layout": value.get("layout", True), "dop": value.get("dop", False)}
    if any(not isinstance(v, bool) for v in result.values()):
        raise ValueError("MOSS 布局验证和 DOP 必须是开关值。")
    return result


def _description(role):
    return " ".join(role.get("description", "").split()).rstrip(".。") or "a person"


def group_prompt(roles, resolution, *, layout=True):
    """Official MOSS schema: center-relative pixel coordinates, [y0, x0, y1, x1]."""
    count = len(roles)
    summary = (f"A full-color image of {count} distinct people sharing one scene, medium-wide shot. "
               "Each person is clearly visible, with separate faces and clothing.")
    descriptions = []
    items = []
    margin = max(1, round(resolution * 0.04))
    for index, role in enumerate(roles):
        position = ("on the left" if index == 0 else "on the right" if index == count - 1 else
                    "in the center" if count == 3 else f"at position {index + 1} from the left")
        caption = f'{role["trigger"]}, {_description(role)}, {position}'
        descriptions.append(caption + ".")
        left = round(margin + index * (resolution - 2 * margin) / count)
        right = round(margin + (index + 1) * (resolution - 2 * margin) / count)
        items.append({"label": "object", "caption": caption,
                      "bbox": [margin - resolution // 2, left - resolution // 2,
                               resolution // 2 - margin, right - resolution // 2]})
    if not layout:
        return summary + " " + " ".join(descriptions)
    return json.dumps({"summary": {"caption": summary}, "items": items,
                       "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"},
                      ensure_ascii=False, separators=(",", ":"))


def validation_prompts(settings, resolution=512):
    resolution = int(resolution)
    if not 64 <= resolution <= 4096:
        raise ValueError("MOSS 布局验证分辨率应在 64～4096 范围内。")
    roles = {r["id"]: r for r in settings["roles"]}
    prompts = [{"kind": "solo", "roles": [r["id"]],
                "prompt": f'A full-color image of {r["trigger"]}, {_description(r)}, medium shot, simple background.'}
               for r in roles.values() if r["trigger"]]
    targets = list(settings["targets"])
    if not targets:
        seen = set()
        for group in settings["groups"]:
            ids = [m["role_id"] for m in group["members"]]
            key = frozenset(ids)
            if len(ids) >= 2 and key not in seen:
                targets.append({"role_ids": ids, "prompt": ""})
                seen.add(key)
        if not targets and len(roles) >= 2:
            ids = list(roles)
            targets.append({"role_ids": ids[:2], "prompt": ""})
            if len(ids) > 2:
                targets.append({"role_ids": ids, "prompt": ""})
    use_layout = options(settings)["layout"]
    for target in targets:
        selected = [roles[rid] for rid in target["role_ids"]]
        if len(selected) < 2:
            continue
        custom = target["prompt"].strip()
        if custom:
            from .multi_character import _contains
            missing = [r["trigger"] for r in selected if not _contains(custom, r["trigger"])]
            if missing:
                raise ValueError("同框验证提示词缺少角色触发词：" + ", ".join(missing))
            lines = prompt_lines(custom)
            if len(lines) != 1:
                raise ValueError("每个目标组合请填写一条验证提示词。")
            prompt = lines[0]
        else:
            prompt = group_prompt(selected, resolution, layout=use_layout)
        prompts.append({"kind": "group", "roles": target["role_ids"], "prompt": prompt})
        if not custom:
            swapped = list(reversed(selected))
            prompts.append({"kind": "group_reversed", "roles": [r["id"] for r in swapped],
                            "prompt": group_prompt(swapped, resolution, layout=use_layout)})
    if options(settings)["dop"]:
        prompts.append({"kind": "no_trigger", "roles": [],
                        "prompt": "A full-color image of two ordinary people talking in a cafe, medium shot."})
    return prompts


def preview_selection(prompts, limit=6):
    """Cover the widest combination, solo identities, a swapped layout and the DOP control."""
    solos = [p for p in prompts if p["kind"] == "solo"]
    groups = [p for p in prompts if p["kind"] == "group"]
    reversed_groups = [p for p in prompts if p["kind"] == "group_reversed"]
    control = [p for p in prompts if p["kind"] == "no_trigger"][:1]
    budget = min(3, len(groups) + len(reversed_groups) + len(control),
                 max(0, limit - min(3, len(solos))))
    ordered = []
    if groups:
        widest = max(groups, key=lambda p: len(p["roles"]))
        ordered = [widest] + [p for p in groups if p is not widest]
        matching = [p for p in reversed_groups if p["roles"] == list(reversed(widest["roles"]))]
        ordered += matching + [p for p in reversed_groups if p not in matching]
    selected_groups = ordered[:max(0, budget - len(control))] + control[:budget]
    return solos[:limit - len(selected_groups)] + selected_groups


def prompt_lines(text, positive=""):
    """Read one natural-language prompt per line or consecutive (also pretty-printed) JSON layouts."""
    decoder = json.JSONDecoder()
    rest = str(text).strip()
    result = []
    while rest:
        if rest.startswith("{"):
            try:
                layout, end = decoder.raw_decode(rest)
            except ValueError as exc:
                raise ValueError("MOSS 布局提示词不是完整 JSON，请检查括号、引号和逗号。") from exc
            summary = layout.get("summary")
            if not isinstance(summary, dict) or not isinstance(summary.get("caption"), str):
                raise ValueError("MOSS 布局提示词需要 summary.caption 场景描述。")
            if not isinstance(layout.get("items"), list):
                raise ValueError("MOSS 布局提示词需要 items 列表。")
            if positive:
                summary["caption"] = positive + ", " + summary["caption"]
            result.append(json.dumps(layout, ensure_ascii=False, separators=(",", ":")))
            rest = rest[end:].strip()
        else:
            line, _, rest = rest.partition("\n")
            line = line.strip()
            if line:
                result.append(", ".join(filter(None, (positive, line))))
            rest = rest.strip()
    return result
