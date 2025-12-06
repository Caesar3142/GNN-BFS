"""
Simple test script to verify OpenFOAM parser works.
"""
from openfoam_parser import OpenFOAMParser
from graph_constructor import GraphConstructor

def test_parser():
    """Test the OpenFOAM parser."""
    data_dir = "BFS-OpenFOAM-data"
    
    print("Testing OpenFOAM parser...")
    parser = OpenFOAMParser(data_dir)
    
    # Test getting time directories
    print("\n1. Getting time directories...")
    time_dirs = parser.get_time_directories()
    print(f"   Found {len(time_dirs)} time directories")
    print(f"   First few: {time_dirs[:5]}")
    
    # Test parsing points
    print("\n2. Parsing mesh points...")
    try:
        points = parser.parse_points()
        print(f"   Successfully parsed {len(points)} points")
        print(f"   Point shape: {points.shape}")
        print(f"   First point: {points[0]}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test parsing connectivity
    print("\n3. Parsing mesh connectivity...")
    try:
        owner, neighbour = parser.parse_connectivity()
        print(f"   Owner array length: {len(owner)}")
        print(f"   Neighbour array length: {len(neighbour)}")
        print(f"   Number of cells: {max(owner.max(), neighbour.max()) + 1}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test parsing fields - use a time directory with nonuniform fields (skip 0 which has uniform)
    if len(time_dirs) > 1:
        test_time = time_dirs[1]  # Use 0.001 instead of 0
        print(f"\n4. Testing field parsing for time: {test_time}")
    elif len(time_dirs) > 0:
        test_time = time_dirs[0]
        print(f"\n4. Testing field parsing for time: {test_time} (may have uniform fields)")
    else:
        print("\n4. No time directories found")
        test_time = None
        
        # Test velocity
        print("   Parsing velocity field...")
        try:
            velocity = parser.parse_vector_field(test_time, "U")
            print(f"   Successfully parsed velocity: shape {velocity.shape}")
            print(f"   Velocity range: [{velocity.min():.3f}, {velocity.max():.3f}]")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test pressure
        print("   Parsing pressure field...")
        try:
            pressure = parser.parse_scalar_field(test_time, "p")
            print(f"   Successfully parsed pressure: shape {pressure.shape}")
            print(f"   Pressure range: [{pressure.min():.3f}, {pressure.max():.3f}]")
        except Exception as e:
            print(f"   Error: {e}")
    
    # Test graph construction
    if test_time:
        print("\n5. Testing graph construction...")
        try:
            graph_constructor = GraphConstructor(parser)
            graph = graph_constructor.build_graph(test_time)
            print(f"   Graph nodes: {graph.x.shape}")
            print(f"   Graph edges: {graph.edge_index.shape}")
            print(f"   Node features: {graph.x.shape[1]} features per node")
        except Exception as e:
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n5. Skipping graph construction test (no time directory)")

if __name__ == "__main__":
    test_parser()

