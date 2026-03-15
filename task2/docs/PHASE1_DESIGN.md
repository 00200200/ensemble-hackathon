# Phase 1 Design: Structural Ingestion & AST Partitioning

## Goal
Transform raw repository into structured, queryable graph and vector database.

## Design Decisions

### 1. Tree-sitter vs Python ast Module

**Decision**: Use Tree-sitter (with Python ast fallback)

**Rationale**:
- Tree-sitter handles incomplete/broken code (critical at cursor positions)
- Supports 100+ languages
- Incremental parsing
- Research shows winning teams use Tree-sitter

**Fallback**: Python native `ast` module if tree-sitter not available

### 2. Entity Types

Primary entities to extract:
- `FunctionDef` / `AsyncFunctionDef`
- `ClassDef`
- `Module` (file-level)

Metadata per entity:
- `file_path`: Absolute path
- `line_range`: (start, end)
- `signature`: Function/class signature
- `raw_code`: Full source code
- `docstring`: Documentation
- `dependencies`: Called functions/classes

### 3. Graph Structure

Using NetworkX DiGraph instead of Neo4j:
- No external setup required
- Faster for batch processing
- Easy serialization with pickle

**Edge Types**:
- `DEFINES`: Module → Function/Class
- `CONTAINS`: Class → Method
- `CALLS`: Function → Function
- `IMPORTS`: Module → Module

### 4. Caching Strategy

MD5-based cache invalidation:
- Hash file content to detect changes
- Cache parsed AST and entities
- Invalidate when file changes

## Implementation Notes

### File Walker
- Exclude: `__pycache__`, `.git`, `venv`, `node_modules`, `*.pyc`
- Follow symlinks: No (to avoid cycles)
- Parallel parsing: Optional (thread pool)

### AST Traversal
- Use Tree-sitter visitor pattern
- Extract call graph via node queries
- Handle both Python 2 and 3 syntax

### Error Handling
- Skip files with syntax errors (log warning)
- Continue processing other files
- Don't crash on single file failure

## Testing Strategy

Unit tests for:
1. Entity extraction from simple functions
2. Class with methods extraction
3. Import statement parsing
4. Call graph construction
5. Cache hit/miss behavior
