import re


class ReAcTreeOutput:

    def __init__(self, op_type, content=None, control=None, subgoals=None):
        self.op_type = op_type      # think / act / expand
        self.content = content      # reasoning or action
        self.control = control      # sequence / fallback / parallel
        self.subgoals = subgoals or []


def parse_response(text: str):
    if not text or not text.strip():
        return ReAcTreeOutput("error", content="empty response")
    
    text = text.strip()
    
    # Check for Expand anywhere in the output (prioritize it)
    if re.search(r"(?i)\bexpand\s*:", text):
        # Extract control flow and subgoals even if "Think:" appears first
        control_match = re.search(r"(?i)(?:control|control flow)\s*:\s*(sequence|fallback|parallel)", text)
        control = control_match.group(1).lower() if control_match else "sequence"
        
        # Extract subgoals from numbered list or inline format
        subgoals = []
        numbered = re.findall(r"\n\s*\d+\.\s*\*?\*?(.+?)(?:\*\*)?(?:\n|$)", text)
        if numbered:
            subgoals = [s.strip(" *:") for s in numbered if s.strip()]
        
        if subgoals:
            return ReAcTreeOutput("expand", control=control, subgoals=subgoals)
    
    # Check for Act/Action anywhere
    action_match = re.search(r"(?i)\b(?:act|action)\s*:\s*(.+?)(?:\n|$)", text)
    if action_match:
        action = action_match.group(1).strip()
        return ReAcTreeOutput("act", content=action)
    
    # Fallback to Think if it starts with Think
    if text.lower().startswith("think"):
        reasoning = re.split(r"(?i)think\s*:\s*", text, maxsplit=1)
        reasoning = reasoning[-1].strip() if len(reasoning) > 1 else text
        return ReAcTreeOutput("think", content=reasoning)
    
    return ReAcTreeOutput("error", content=text)