"""
Graph Store for Code Relationships

Stores entities and their relationships in a graph structure.
Uses NetworkX DiGraph for simplicity and performance.
"""

from __future__ import annotations

import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    # Simple fallback graph implementation
    class SimpleDiGraph:
        """Simple directed graph fallback when NetworkX not available."""
        
        def __init__(self):
            self._nodes: Dict[str, Dict[str, Any]] = {}
            self._edges: Dict[str, List[Tuple[str, str, Dict[str, Any]]]] = defaultdict(list)
            self._incoming: Dict[str, List[str]] = defaultdict(list)
        
        def add_node(self, node_id: str, **attrs) -> None:
            if node_id not in self._nodes:
                self._nodes[node_id] = {}
            self._nodes[node_id].update(attrs)
        
        def add_edge(self, source: str, target: str, **attrs) -> None:
            self._edges[source].append((source, target, attrs))
            self._incoming[target].append(source)
            # Ensure nodes exist
            if source not in self._nodes:
                self.add_node(source)
            if target not in self._nodes:
                self.add_node(target)
        
        def has_node(self, node_id: str) -> bool:
            return node_id in self._nodes
        
        def has_edge(self, source: str, target: str) -> bool:
            if source not in self._edges:
                return False
            return any(e[1] == target for e in self._edges[source])
        
        def nodes(self):
            return list(self._nodes.keys())
        
        def edges(self, node_id: str = None):
            if node_id:
                return [(e[0], e[1]) for e in self._edges.get(node_id, [])]
            all_edges = []
            for edges in self._edges.values():
                all_edges.extend([(e[0], e[1]) for e in edges])
            return all_edges
        
        def neighbors(self, node_id: str) -> List[str]:
            """Get outgoing neighbors."""
            return [e[1] for e in self._edges.get(node_id, [])]
        
        def predecessors(self, node_id: str) -> List[str]:
            """Get incoming neighbors."""
            return self._incoming.get(node_id, [])
        
        def degree(self, node_id: str) -> int:
            """Get total degree (in + out)."""
            out_deg = len(self._edges.get(node_id, []))
            in_deg = len(self._incoming.get(node_id, []))
            return out_deg + in_deg
        
        def get_node_data(self, node_id: str) -> Optional[Dict[str, Any]]:
            return self._nodes.get(node_id)
        
        def get_edge_data(self, source: str, target: str) -> Optional[Dict[str, Any]]:
            for e in self._edges.get(source, []):
                if e[1] == target:
                    return e[2]
            return None

logger = logging.getLogger(__name__)


class CodeGraph:
    """Graph store for code entities and relationships."""
    
    def __init__(self):
        if NETWORKX_AVAILABLE:
            self.graph = nx.DiGraph()
        else:
            logger.warning("NetworkX not available, using simple graph fallback")
            self.graph = SimpleDiGraph()
        
        self._file_to_entities: Dict[str, Set[str]] = defaultdict(set)
        self._name_to_entities: Dict[str, Set[str]] = defaultdict(set)
    
    def add_entity(self, entity, entity_type: str = "entity") -> str:
        """
        Add an entity to the graph.
        
        Returns:
            Node ID
        """
        node_id = entity.id if hasattr(entity, 'id') else str(entity)
        
        # Add node with attributes
        self.graph.add_node(
            node_id,
            entity=entity,
            type=entity_type,
            name=getattr(entity, 'name', str(entity)),
            file_path=getattr(entity, 'file_path', ''),
            line_start=getattr(entity, 'line_start', 0),
            entity_type=getattr(entity, 'type', 'unknown')
        )
        
        # Index by file
        file_path = getattr(entity, 'file_path', None)
        if file_path:
            self._file_to_entities[file_path].add(node_id)
        
        # Index by name
        name = getattr(entity, 'name', None)
        if name:
            self._name_to_entities[name].add(node_id)
        
        return node_id
    
    def add_relationship(
        self, 
        source_id: str, 
        target_id: str, 
        relation_type: str,
        **attrs
    ) -> None:
        """Add a relationship edge between two entities."""
        if not self.graph.has_node(source_id):
            logger.warning(f"Source node not found: {source_id}")
            return
        if not self.graph.has_node(target_id):
            # Don't warn for external modules (they're expected to not exist)
            # Only warn for internal entity references
            if not target_id.startswith('module:'):
                logger.warning(f"Target node not found: {target_id}")
            return
        
        self.graph.add_edge(
            source_id, 
            target_id, 
            relation=relation_type,
            **attrs
        )
    
    def add_file_entities(
        self,
        file_path: str,
        entities: List[Any],
        imports: List[Dict[str, Any]]
    ) -> None:
        """Add all entities from a file and their relationships."""
        # Add module node
        module_id = f"{file_path}:module"
        self.graph.add_node(
            module_id,
            type='module',
            file_path=file_path,
            name=Path(file_path).name
        )
        self._file_to_entities[file_path].add(module_id)
        
        # Add entities and connect to module
        entity_ids = []
        for entity in entities:
            entity_id = self.add_entity(entity, entity.type)
            entity_ids.append(entity_id)
            
            # Add DEFINES relationship: module -> entity
            self.add_relationship(module_id, entity_id, 'DEFINES')
            
            # Add CONTAINS relationship: class -> method
            if hasattr(entity, 'parent') and entity.parent:
                parent_id = f"{file_path}:{entity.parent}"
                if self.graph.has_node(parent_id):
                    self.add_relationship(parent_id, entity_id, 'CONTAINS')
            
            # Add CALLS relationships
            if hasattr(entity, 'dependencies') and entity.dependencies:
                for dep in entity.dependencies:
                    # Try to find the dependency in known entities
                    dep_id = self._resolve_dependency(file_path, dep)
                    if dep_id:
                        self.add_relationship(entity_id, dep_id, 'CALLS')
        
        # Add IMPORTS relationships
        for imp in imports:
            if imp.get('type') == 'from':
                module = imp.get('module', '')
                if module:
                    import_id = f"module:{module}"
                    self.add_relationship(module_id, import_id, 'IMPORTS')
    
    def _resolve_dependency(self, current_file: str, dep_name: str) -> Optional[str]:
        """Try to resolve a dependency name to an entity ID."""
        # Direct match
        if dep_name in self._name_to_entities:
            candidates = self._name_to_entities[dep_name]
            # Prefer same file
            for cand in candidates:
                if NETWORKX_AVAILABLE:
                    data = self.graph.nodes.get(cand)
                else:
                    data = self.graph.get_node_data(cand)
                if data and data.get('file_path') == current_file:
                    return cand
            # Return first match
            return next(iter(candidates))
        
        # Try with file prefix
        file_prefix = f"{current_file}:"
        full_name = file_prefix + dep_name
        if full_name in self._name_to_entities.get(dep_name, set()):
            return full_name
        
        return None
    
    def get_entity(self, entity_id: str) -> Optional[Any]:
        """Get entity by ID."""
        if not self.graph.has_node(entity_id):
            return None
        if NETWORKX_AVAILABLE:
            data = self.graph.nodes.get(entity_id)
        else:
            data = self.graph.get_node_data(entity_id)
        return data.get('entity') if data else None
    
    def get_callees(self, entity_id: str) -> List[Tuple[str, Any]]:
        """Get entities called by this entity (outgoing edges)."""
        if not self.graph.has_node(entity_id):
            return []
        
        results = []
        if NETWORKX_AVAILABLE:
            for target in self.graph.successors(entity_id):
                edge_data = self.graph.get_edge_data(entity_id, target)
                if edge_data and edge_data.get('relation') == 'CALLS':
                    entity = self.get_entity(target)
                    if entity:
                        results.append((target, entity))
        else:
            for target in self.graph.neighbors(entity_id):
                edge_data = self.graph.get_edge_data(entity_id, target)
                if edge_data and edge_data.get('relation') == 'CALLS':
                    entity = self.get_entity(target)
                    if entity:
                        results.append((target, entity))
        
        return results
    
    def get_callers(self, entity_id: str) -> List[Tuple[str, Any]]:
        """Get entities that call this entity (incoming edges)."""
        if not self.graph.has_node(entity_id):
            return []
        
        results = []
        if NETWORKX_AVAILABLE:
            for source in self.graph.predecessors(entity_id):
                edge_data = self.graph.get_edge_data(source, entity_id)
                if edge_data and edge_data.get('relation') == 'CALLS':
                    entity = self.get_entity(source)
                    if entity:
                        results.append((source, entity))
        else:
            for source in self.graph.predecessors(entity_id):
                edge_data = self.graph.get_edge_data(source, entity_id)
                if edge_data and edge_data.get('relation') == 'CALLS':
                    entity = self.get_entity(source)
                    if entity:
                        results.append((source, entity))
        
        return results
    
    def get_neighbors(
        self, 
        entity_id: str, 
        relation_types: Optional[List[str]] = None,
        depth: int = 1
    ) -> List[Tuple[str, Any, int]]:
        """
        Get neighboring entities up to a certain depth.
        
        Returns:
            List of (entity_id, entity, depth) tuples
        """
        if not self.graph.has_node(entity_id):
            return []
        
        visited = {entity_id: 0}
        queue = [(entity_id, 0)]
        results = []
        
        while queue:
            current_id, current_depth = queue.pop(0)
            
            if current_depth >= depth:
                continue
            
            # Get neighbors
            neighbors = []
            if NETWORKX_AVAILABLE:
                for succ in self.graph.successors(current_id):
                    if current_depth + 1 <= depth and succ not in visited:
                        edge_data = self.graph.get_edge_data(current_id, succ)
                        if edge_data:
                            rel_type = edge_data.get('relation')
                            if relation_types is None or rel_type in relation_types:
                                neighbors.append(succ)
                
                for pred in self.graph.predecessors(current_id):
                    if current_depth + 1 <= depth and pred not in visited:
                        edge_data = self.graph.get_edge_data(pred, current_id)
                        if edge_data:
                            rel_type = edge_data.get('relation')
                            if relation_types is None or rel_type in relation_types:
                                neighbors.append(pred)
            else:
                for succ in self.graph.neighbors(current_id):
                    if current_depth + 1 <= depth and succ not in visited:
                        edge_data = self.graph.get_edge_data(current_id, succ)
                        if edge_data:
                            rel_type = edge_data.get('relation')
                            if relation_types is None or rel_type in relation_types:
                                neighbors.append(succ)
                
                for pred in self.graph.predecessors(current_id):
                    if current_depth + 1 <= depth and pred not in visited:
                        edge_data = self.graph.get_edge_data(pred, current_id)
                        if edge_data:
                            rel_type = edge_data.get('relation')
                            if relation_types is None or rel_type in relation_types:
                                neighbors.append(pred)
            
            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    visited[neighbor_id] = current_depth + 1
                    entity = self.get_entity(neighbor_id)
                    if entity:
                        results.append((neighbor_id, entity, current_depth + 1))
                    queue.append((neighbor_id, current_depth + 1))
        
        return results
    
    def get_degree(self, entity_id: str) -> int:
        """Get the degree (number of connections) of an entity."""
        if not self.graph.has_node(entity_id):
            return 0
        return self.graph.degree(entity_id)
    
    def is_high_degree(self, entity_id: str, threshold: int = 20) -> bool:
        """Check if entity has high degree (utility function pattern)."""
        return self.get_degree(entity_id) > threshold
    
    def get_entities_by_file(self, file_path: str) -> List[Any]:
        """Get all entities from a file."""
        entity_ids = self._file_to_entities.get(file_path, set())
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"get_entities_by_file: {file_path} -> {len(entity_ids)} entity IDs")
        if not entity_ids and file_path in str(self._file_to_entities.keys())[:200]:
            logger.debug(f"  Available paths: {list(self._file_to_entities.keys())[:10]}")
        return [self.get_entity(eid) for eid in entity_ids if self.get_entity(eid)]
    
    def get_entities_by_name(self, name: str) -> List[Any]:
        """Get all entities with a given name."""
        entity_ids = self._name_to_entities.get(name, set())
        return [self.get_entity(eid) for eid in entity_ids if self.get_entity(eid)]
    
    def get_all_entities(self) -> List[Tuple[str, Any]]:
        """Get all entities in the graph."""
        results = []
        for node_id in self.graph.nodes():
            entity = self.get_entity(node_id)
            if entity:
                results.append((node_id, entity))
        return results
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the graph to a file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'graph': self.graph,
            'file_to_entities': dict(self._file_to_entities),
            'name_to_entities': dict(self._name_to_entities)
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Graph saved to {path}")
    
    def load(self, path: Union[str, Path]) -> None:
        """Load the graph from a file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Graph file not found: {path}")
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.graph = data['graph']
        self._file_to_entities = defaultdict(set, data['file_to_entities'])
        self._name_to_entities = defaultdict(set, data['name_to_entities'])
        
        logger.info(f"Graph loaded from {path}")
    
    def __len__(self) -> int:
        """Return number of entities in the graph."""
        return len(list(self.graph.nodes()))
