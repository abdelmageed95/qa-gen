"""Create detailed LangGraph workflow visualization with metadata"""

from app.services.agent_workflow import MultiAgentWorkflow
from app.services.vector_store import VectorStore
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Initialize the workflow
vector_store = VectorStore()
workflow_obj = MultiAgentWorkflow(vector_store=vector_store)

# Get the graph
graph = workflow_obj.workflow.get_graph()

# Get graph nodes and edges
nodes = graph.nodes
edges = graph.edges

print("Graph Nodes:")
for node_id, node_data in nodes.items():
    print(f"  - {node_id}: {node_data}")

print("\nGraph Edges:")
for edge in edges:
    print(f"  - {edge}")

# Create detailed visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

# Left side: Original graph
ax1.axis('off')
ax1.text(0.5, 0.95, 'LangGraph Structure (from get_graph())',
         fontsize=14, weight='bold', ha='center', transform=ax1.transAxes,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD',
                  edgecolor='#1976D2', linewidth=2))

# Display the original PNG
try:
    img = plt.imread('langgraph_workflow.png')
    ax1.imshow(img)
except:
    ax1.text(0.5, 0.5, 'Original graph loaded from get_graph()',
             ha='center', va='center', transform=ax1.transAxes)

# Right side: Detailed information
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 14)
ax2.axis('off')

ax2.text(5, 13, 'Workflow Details',
         fontsize=14, weight='bold', ha='center',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF3E0',
                  edgecolor='#F57C00', linewidth=2))

# Node information
y_pos = 11.5
ax2.text(5, y_pos, 'Nodes in Graph:',
         fontsize=12, weight='bold', ha='center')

y_pos -= 0.6
node_info = []
for node_id in nodes.keys():
    node_info.append(f"• {node_id}")

ax2.text(5, y_pos, '\n'.join(node_info),
         fontsize=10, ha='center', va='top',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9',
                  edgecolor='#4CAF50', linewidth=1.5, alpha=0.8))

# Edge information
y_pos -= len(node_info) * 0.35 + 0.8
ax2.text(5, y_pos, 'Edges (Transitions):',
         fontsize=12, weight='bold', ha='center')

y_pos -= 0.6
edge_info = []
for edge in edges:
    if hasattr(edge, 'source') and hasattr(edge, 'target'):
        edge_info.append(f"{edge.source} → {edge.target}")
    else:
        edge_info.append(str(edge))

ax2.text(5, y_pos, '\n'.join(edge_info[:10]),  # Limit to first 10 edges
         fontsize=9, ha='center', va='top', family='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9C4',
                  edgecolor='#F57F17', linewidth=1.5, alpha=0.8))

# Workflow characteristics
y_pos -= len(edge_info[:10]) * 0.3 + 0.8
characteristics = f"""Workflow Characteristics:
━━━━━━━━━━━━━━━━━━━━━━━━━━
Type: StateGraph
Compiled: Yes
Entry Point: __start__
Exit Points: __end__
Conditional Edges: 1
Feedback Loops: 2"""

ax2.text(5, y_pos, characteristics,
         fontsize=9, ha='center', va='top', family='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#E1F5FE',
                  edgecolor='#0277BD', linewidth=1.5, alpha=0.8))

# State schema
y_pos -= 3.2
state_schema = f"""State Schema:
━━━━━━━━━━━━━━━━━━━━━━━━━━
• document_id: str
• query: str
• num_questions: int
• context: str
• generated_questions: List
• evaluation_results: List
• iteration: int
• max_iterations: int
• feedback_history: List
• approved: bool
• quality_warnings: List"""

ax2.text(5, y_pos, state_schema,
         fontsize=8, ha='center', va='top', family='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#F3E5F5',
                  edgecolor='#7B1FA2', linewidth=1.5, alpha=0.8))

plt.tight_layout()
plt.savefig('langgraph_workflow_detailed.png', dpi=300,
           bbox_inches='tight', facecolor='white')
print("\nDetailed workflow visualization saved as 'langgraph_workflow_detailed.png'")
plt.close()

# Also save the graph structure as text
with open('langgraph_workflow_structure.txt', 'w') as f:
    f.write("LangGraph Workflow Structure\n")
    f.write("=" * 50 + "\n\n")

    f.write("NODES:\n")
    f.write("-" * 50 + "\n")
    for node_id, node_data in nodes.items():
        f.write(f"{node_id}\n")
        if node_data:
            f.write(f"  Data: {node_data}\n")
        f.write("\n")

    f.write("\nEDGES:\n")
    f.write("-" * 50 + "\n")
    for edge in edges:
        f.write(f"{edge}\n")

    f.write("\n\nWORKFLOW EXECUTION ORDER:\n")
    f.write("-" * 50 + "\n")
    f.write("1. __start__ → retrieve_context\n")
    f.write("2. retrieve_context → generate_questions\n")
    f.write("3. generate_questions → evaluate_questions\n")
    f.write("4. evaluate_questions → [DECISION]\n")
    f.write("   - If approved: → __end__\n")
    f.write("   - If not approved: → regenerate_questions\n")
    f.write("5. regenerate_questions → evaluate_questions (loop)\n")

print("Graph structure saved as 'langgraph_workflow_structure.txt'")
