from src.recourse.graph_builder import RecourseGraphBuilder
import inspect

builder = RecourseGraphBuilder(None, None, None)
print(f"Methods in RecourseGraphBuilder: {[m for m in dir(builder) if not m.startswith('__')]}")
if hasattr(builder, 'find_greedy_path'):
    print("SUCCESS: find_greedy_path exists.")
else:
    print("FAILURE: find_greedy_path missing.")

print(f"File source: {inspect.getfile(RecourseGraphBuilder)}")
