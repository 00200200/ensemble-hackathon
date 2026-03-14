from collections import defaultdict, deque

import numpy as np

N_CLASSES = 500
CLASS_COLS = [f"class_{i}" for i in range(N_CLASSES)]


def parse_obo(path: str) -> dict[str, list[str]]:
    parents: dict[str, list[str]] = {}
    current_id = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("id: "):
                current_id = line[4:].split()[0]
            elif line.startswith("is_a: ") and current_id:
                parent = line[6:].split()[0]
                parents.setdefault(current_id, []).append(parent)
    return parents


def build_children_map(parents: dict[str, list[str]]) -> dict[int, list[int]]:
    col_to_idx = {c: i for i, c in enumerate(CLASS_COLS)}
    valid = set(CLASS_COLS)
    children: dict[int, list[int]] = defaultdict(list)

    for child, plist in parents.items():
        if child not in valid:
            continue
        for p in plist:
            if p in valid:
                children[col_to_idx[p]].append(col_to_idx[child])

    return dict(children)


def topological_order(children_map: dict[int, list[int]]) -> list[int]:
    in_degree = [0] * N_CLASSES
    for child_list in children_map.values():
        for c in child_list:
            in_degree[c] += 1

    queue = deque(i for i in range(N_CLASSES) if in_degree[i] == 0)
    order: list[int] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in children_map.get(node, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    return order


def enforce_hierarchy(
    arr: np.ndarray,
    children_map: dict[int, list[int]],
    topo_order: list[int],
) -> np.ndarray:
    out = arr.copy()
    for node in reversed(topo_order):
        for child in children_map.get(node, []):
            out[:, node] = np.maximum(out[:, node], out[:, child])
    return out
