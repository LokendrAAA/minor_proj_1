import re

def calculate_cpp_metrics(code):
    # Count Source Lines of Code (SLOC)
    sloc = len([line for line in code.splitlines() if line.strip() and not line.strip().startswith("//")])

    # Calculate Cyclomatic Complexity
    complexity_keywords = ['if', 'else if', 'while', 'for', 'case', 'catch']
    cyclomatic_complexity = 1 + sum(len(re.findall(r'\b' + kw + r'\b', code)) for kw in complexity_keywords)

    # Inheritance Tree Depth (Dummy Example)
    inheritance_matches = re.findall(r'class\s+\w+\s*:\s*public\s+\w+', code)
    inheritance_tree_depth = len(inheritance_matches)

    # Afferent and Efferent Coupling
    afferent_coupling = len(re.findall(r'class\s+\w+', code))
    efferent_coupling = len(re.findall(r'#include\s+<.*?>', code))

    # Instability
    instability = efferent_coupling / (afferent_coupling + efferent_coupling) if (afferent_coupling + efferent_coupling) > 0 else 0

    # Distance (Assumed Abstractness is 0.5 for simplicity)
    abstractness = 0.5
    distance = abs(abstractness + instability - 1)

    # Return metrics as a dictionary
    return {
        "SLOC": sloc,
        "Cyclomatic Complexity": cyclomatic_complexity,
        "Inheritance Tree Depth": inheritance_tree_depth,
        "Afferent Coupling": afferent_coupling,
        "Efferent Coupling": efferent_coupling,
        "Instability": round(instability, 2),
        "Distance": round(distance, 2)
    }
