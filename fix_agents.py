#!/usr/bin/env python3
"""Script to update all agents to use the enhanced base class."""

import re
from pathlib import Path

# Agent files and their output types
AGENTS_TO_FIX = {
    "compression.py": ("CompressionAgent", "CompressedFindings"),
    "query_transformation.py": ("QueryTransformationAgent", "TransformedQuery"),
    "report_generator.py": ("ReportGeneratorAgent", "ResearchReport"),
    "research_executor.py": ("ResearchExecutorAgent", "ResearchFindings"),
}

def fix_agent_file(file_path: Path, class_name: str, output_type: str):
    """Fix an agent file to use the enhanced base class."""
    content = file_path.read_text()
    
    # Check if already fixed
    if "_get_output_type" in content:
        print(f"✅ {file_path.name} already fixed")
        return
    
    # Add AgentConfiguration import if not present
    if "AgentConfiguration" not in content:
        content = content.replace(
            "from open_deep_research_with_pydantic_ai.agents.base import (",
            "from open_deep_research_with_pydantic_ai.agents.base import (\n    AgentConfiguration,"
        )
    
    # Find and fix the __init__ method
    init_pattern = r'(    def __init__\(self[^)]*\):\s*\n.*?"""[^"]*"""\s*\n\s*super\(\).__init__\([^)]+\))'
    
    def replace_init(match):
        indent = "        "
        agent_type = class_name.lower().replace("agent", "")
        new_init = f'''    def __init__(self):
        """Initialize the {agent_type} agent."""
        config = AgentConfiguration(
            agent_name="{agent_type}_agent",
            agent_type="{agent_type}",
        )
        super().__init__(config=config)'''
        return new_init
    
    content = re.sub(init_pattern, replace_init, content, flags=re.DOTALL)
    
    # Add _get_output_type method after __init__
    # Find the end of __init__ method and add the new method
    lines = content.split('\n')
    new_lines = []
    in_init = False
    init_done = False
    
    for i, line in enumerate(lines):
        new_lines.append(line)
        
        if "def __init__(self)" in line:
            in_init = True
        elif in_init and not init_done and (line.strip().startswith("def ") or (i > 0 and not line.strip() and i < len(lines) - 1 and lines[i+1].strip().startswith("def "))):
            # Found the next method after __init__
            if not line.strip().startswith("def _get_output_type"):
                # Insert the _get_output_type method
                new_lines.insert(-1, f'''
    def _get_output_type(self) -> type[{output_type}]:
        """Get the output type for this agent."""
        return {output_type}''')
            init_done = True
            in_init = False
    
    content = '\n'.join(new_lines)
    
    # Write back the fixed content
    file_path.write_text(content)
    print(f"✅ Fixed {file_path.name}")

def main():
    """Fix all agent files."""
    agents_dir = Path("/Users/keng/oss/open_deep_research_pydantic_ai/src/open_deep_research_with_pydantic_ai/agents")
    
    for filename, (class_name, output_type) in AGENTS_TO_FIX.items():
        file_path = agents_dir / filename
        if file_path.exists():
            try:
                fix_agent_file(file_path, class_name, output_type)
            except Exception as e:
                print(f"❌ Error fixing {filename}: {e}")
        else:
            print(f"❌ File not found: {filename}")

if __name__ == "__main__":
    main()