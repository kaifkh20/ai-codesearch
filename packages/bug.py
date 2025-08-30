def rule_try_except(code,issues=[]):
    if "try:" not in code:
        issues.append("No error handling detected")
    
    return issues

def rules_bug(code):
    issues = []
    issues = rule_try_except(code,issues)

    return issues
