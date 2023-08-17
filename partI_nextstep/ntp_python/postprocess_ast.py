# A naive post-processing of the .ast.json file for instructional purposes.
# Please see LeanDojo's `TracedFile` and `FileNode4` classes for a complete version.

def reconstruct_theorem_proof(command_ast):
    pieces = []
    pos = []
    endPos = []
    def _process_arg(node_arg):
        if 'atom' in node_arg:
            pieces.append(node_arg['atom']['info']['original']['leading'])
            pieces.append(node_arg['atom']['val'])
            pieces.append(node_arg['atom']['info']['original']['trailing'])
            pos.append(node_arg['atom']['info']['original']['pos'])
            endPos.append(node_arg['atom']['info']['original']['endPos'])
        if 'ident' in node_arg:
            pieces.append(node_arg['ident']['info']['original']['leading'])
            pieces.append(node_arg['ident']['rawVal'])
            pieces.append(node_arg['ident']['info']['original']['trailing'])
            pos.append(node_arg['ident']['info']['original']['pos'])
            endPos.append(node_arg['ident']['info']['original']['endPos'])
        if 'node' in node_arg:
            _process_node(node_arg['node'])
        
    def _process_node(node):
        if 'args' in node:
            for arg in node['args']:
                _process_arg(arg)
    
    _process_node(command_ast['node'])
    out = ''.join(pieces)
    start = min(pos)
    end = max(endPos)
    return start, end, out

def get_theorem(start, ast):
    for command in ast['commandASTs']:
        start_, end_, thm = reconstruct_theorem_proof(command)
        if start_ <= start <= end_:
            return thm