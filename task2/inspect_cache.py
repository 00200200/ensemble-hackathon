#!/usr/bin/env python3
"""Inspect cached repository data for debugging."""

import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

def main():
    cache_dir = Path("cache")
    
    print("=== Cache Inspection ===\n")
    
    # List cache files
    print("Cache files:")
    for f in cache_dir.iterdir():
        print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    
    print("\n" + "="*50 + "\n")
    
    # Load and inspect graph
    graph_files = list(cache_dir.glob("*_graph.pkl"))
    if graph_files:
        print(f"Loading graph: {graph_files[0]}")
        from task2.graph_store import CodeGraph
        graph = CodeGraph.load(graph_files[0])
        
        stats = graph.get_stats()
        print(f"Graph stats: {stats}")
        
        entities = graph.get_all_entities()
        print(f"\nSample entities:")
        for e in entities[:3]:
            print(f"  - {e.name} ({e.type}) in {e.file_path}")
    else:
        print("No graph files found")
    
    print("\n" + "="*50 + "\n")
    
    # Load and inspect retriever
    retriever_dirs = [d for d in cache_dir.iterdir() if d.is_dir()]
    if retriever_dirs:
        print(f"Loading retriever: {retriever_dirs[0]}")
        from task2.vector_store import HybridRetriever
        retriever = HybridRetriever()
        retriever.load(retriever_dirs[0])
        
        print(f"Number of indexed entities: {len(retriever.entity_ids)}")
        
        # Test search
        if retriever.entity_ids:
            print("\nTest search for 'function':")
            results = retriever.search("function", k=3)
            for r in results:
                print(f"  - {r.entity.name}: {r.score:.3f}")
    else:
        print("No retriever directories found")

if __name__ == "__main__":
    main()
