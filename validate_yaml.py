import yaml

yaml_path = "metadata/meta.yaml"

print("🔍 Validating YAML file...")

try:
    with open(yaml_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    print("✅ YAML is valid!")
    
except yaml.YAMLError as e:
    print(f"❌ YAML Error: {e}")
    print("\n" + "="*60)
    print("COMMON YAML ISSUES TO CHECK:")
    print("="*60)
    
    # Read the file to check for issues
    with open(yaml_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Find problem area
    if hasattr(e, 'problem_mark'):
        mark = e.problem_mark
        problem_line = mark.line + 1
        problem_col = mark.column + 1
        
        print(f"\n📍 Problem near line {problem_line}, column {problem_col}")
        print("\nContext (5 lines before and after):\n")
        
        start = max(0, problem_line - 6)
        end = min(len(lines), problem_line + 5)
        
        for i in range(start, end):
            prefix = ">>> " if i == problem_line - 1 else "    "
            print(f"{prefix}Line {i+1}: {lines[i]}", end="")
    
    print("\n" + "="*60)
    print("COMMON FIXES:")
    print("="*60)
    print("1. Check if all list items start with '-' (dash)")
    print("2. Ensure consistent indentation (use spaces, not tabs)")
    print("3. Multi-line strings should use quotes or '|' or '>'")
    print("4. Colons in values must be quoted: \"text: with colon\"")
    print("\nExample correct format:")
    print("  rules:")
    print("    - \"First rule\"")
    print("    - \"Second rule\"")
    print("    - \"Rule with: colon must be quoted\"")