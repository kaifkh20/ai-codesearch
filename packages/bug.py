#THIS FILE IS GENERATED USING LLM(Lazy to do regex)

import re

def rule_try_except(code, issues=None):
    """Check for proper error handling"""
    if issues is None:
        issues = []
    
    if "try:" not in code:
        issues.append("No error handling detected - consider using try/except blocks")
    
    # Check for bare except clauses
    if re.search(r'except\s*:', code):
        issues.append("Bare except clause detected - specify exception types")
    
    return issues

def rule_eval_exec(code, issues=None):
    """Check for dangerous eval/exec usage"""
    if issues is None:
        issues = []
    
    pattern = r'\b(eval|exec)\s*\('
    matches = re.findall(pattern, code)
    if matches:
        issues.append(f"Detected use of {'/'.join(set(matches))} - potential code injection risk")
    
    return issues

def rule_sql_injection(code, issues=None):
    """Check for potential SQL injection vulnerabilities"""
    if issues is None:
        issues = []
    
    # Look for string formatting in SQL-like statements
    sql_patterns = [
        r'(SELECT|INSERT|UPDATE|DELETE).*%\w',
        r'(SELECT|INSERT|UPDATE|DELETE).*\+.*\w',
        r'(SELECT|INSERT|UPDATE|DELETE).*\.format\(',
        r'(SELECT|INSERT|UPDATE|DELETE).*f["\']'
    ]
    
    for pattern in sql_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            issues.append("Potential SQL injection vulnerability - use parameterized queries")
            break
    
    return issues

def rule_hardcoded_secrets(code, issues=None):
    """Check for hardcoded passwords, keys, tokens"""
    if issues is None:
        issues = []
    
    secret_patterns = [
        r'password\s*=\s*["\'][^"\']+["\']',
        r'api[_-]?key\s*=\s*["\'][^"\']+["\']',
        r'secret\s*=\s*["\'][^"\']+["\']',
        r'token\s*=\s*["\'][^"\']+["\']',
        r'auth[_-]?key\s*=\s*["\'][^"\']+["\']'
    ]
    
    for pattern in secret_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            issues.append("Hardcoded credentials detected - use environment variables or secure vaults")
            break
    
    return issues

def rule_unsafe_imports(code, issues=None):
    """Check for potentially unsafe imports"""
    if issues is None:
        issues = []
    
    unsafe_modules = ['pickle', 'subprocess', 'os.system', 'commands']
    
    for module in unsafe_modules:
        if re.search(rf'\bimport\s+{module}\b', code) or re.search(rf'\bfrom\s+{module}\s+import\b', code):
            issues.append(f"Potentially unsafe import: {module} - review for security implications")
    
    return issues

def rule_file_operations(code, issues=None):
    """Check for unsafe file operations"""
    if issues is None:
        issues = []
    
    # Check for path traversal vulnerabilities
    if re.search(r'open\s*\([^)]*\.\.[/\\]', code):
        issues.append("Potential path traversal vulnerability detected")
    
    # Check for unsafe file permissions
    if re.search(r'chmod\s*\([^)]*0o777', code) or re.search(r'chmod\s*\([^)]*777', code):
        issues.append("Overly permissive file permissions (777) detected")
    
    return issues

def rule_input_validation(code, issues=None):
    """Check for missing input validation"""
    if issues is None:
        issues = []
    
    # Check for direct use of input() without validation
    if re.search(r'\binput\s*\(', code) and 'int(' not in code and 'len(' not in code:
        issues.append("Direct use of input() without validation - consider adding input sanitization")
    
    return issues

def rule_debug_code(code, issues=None):
    """Check for debug code that shouldn't be in production"""
    if issues is None:
        issues = []
    
    debug_patterns = [
        r'\bdebugger\b',
        r'console\.log\(',
        r'#\s*TODO',
        r'#\s*FIXME',
        r'#\s*HACK'
    ]
    
    for pattern in debug_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            issues.append("Debug/development code detected - remove before production")
            break
    
    return issues

def rule_weak_crypto(code, issues=None):
    """Check for weak cryptographic practices"""
    if issues is None:
        issues = []
    
    weak_crypto_patterns = [
        r'\bmd5\b',
        r'\bsha1\b',
        r'Random\(\)',
        r'random\.random\(\)'
    ]
    
    for pattern in weak_crypto_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            issues.append("Weak cryptographic function detected - use stronger alternatives")
            break
    
    return issues

def rule_buffer_overflow(code, issues=None):
    """Check for potential buffer overflow issues (mainly for C-style operations)"""
    if issues is None:
        issues = []
    
    # In Python context, check for large allocations without bounds checking
    if re.search(r'\[\s*\d{6,}\s*\]', code):  # Arrays with very large sizes
        issues.append("Large fixed-size allocation detected - consider bounds checking")
    
    return issues

def rule_race_conditions(code, issues=None):
    """Check for potential race condition vulnerabilities"""
    if issues is None:
        issues = []
    
    # Check for file operations without proper locking
    if ('open(' in code and 'threading' in code) and 'lock' not in code.lower():
        issues.append("File operations in threaded code without locking - potential race condition")
    
    return issues

def rules_bug(code):
    """Main function to run all security checks"""
    issues = []
    
    # Run all rule checks
    issues = rule_try_except(code, issues)
    issues = rule_eval_exec(code, issues)
    issues = rule_sql_injection(code, issues)
    issues = rule_hardcoded_secrets(code, issues)
    issues = rule_unsafe_imports(code, issues)
    issues = rule_file_operations(code, issues)
    issues = rule_input_validation(code, issues)
    issues = rule_debug_code(code, issues)
    issues = rule_weak_crypto(code, issues)
    issues = rule_buffer_overflow(code, issues)
    issues = rule_race_conditions(code, issues)
    
    return issues
