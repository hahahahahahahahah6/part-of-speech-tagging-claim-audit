"""Deterministic local source and documentation inspection."""
import ast
from pathlib import Path
import re
from .phrases import POS, NOPOS, GENERIC_NAMES

POS_RULE = 'documented-part-of-speech-tagging-claim-false'
NO_RULE = 'documented-no-part-of-speech-tagging-claim-false'


def phrase_matches(text, phrases):
    text = text.lower()
    found = []
    for phrase in sorted(phrases, key=len, reverse=True):
        start = text.find(phrase)
        while start >= 0:
            found.append(phrase)
            text = text[:start] + ' ' * len(phrase) + text[start + len(phrase):]
            start = text.find(phrase)
    return found, text


def prose(text):
    lines = []
    fenced = False
    for line in text.splitlines():
        if line.lstrip().startswith('```'):
            fenced = not fenced
        elif not fenced:
            lines.append(line.replace('`', ''))
    return '\n'.join(lines)


def has_evidence(source):
    tree = ast.parse(source)
    shadow = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    imports = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                imports[item.asname or item.name.split('.')[0]] = item.name if item.asname else item.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            for item in node.names:
                imports[item.asname or item.name] = node.module + '.' + item.name
    for name in shadow:
        imports.pop(name, None)

    def resolve(node):
        if isinstance(node, ast.Name):
            return imports.get(node.id, '')
        if isinstance(node, ast.Attribute):
            base = resolve(node.value)
            return base + '.' + node.attr if base else ''
        return ''

    values = {}
    def literal(node):
        return node.value.lower() if isinstance(node, ast.Constant) and isinstance(node.value, str) else ''

    def value(node):
        if isinstance(node, ast.Name):
            return values.get(node.id, '') if node.id not in shadow else ''
        if isinstance(node, ast.Call):
            route = resolve(node.func)
            if route == 'spacy.load':
                return 'spacy-model'
            if route == 'stanza.Pipeline' and any(k.arg == 'processors' and 'pos' in literal(k.value) for k in node.keywords):
                return 'stanza-model'
            if route.startswith('flair.') and route.endswith('.SequenceTagger.load') and node.args and 'pos' in literal(node.args[0]):
                return 'flair-model'
            base = value(node.func)
            if base in ('spacy-model', 'stanza-model'):
                return base.replace('model', 'result')
        return ''

    nodes = list(ast.walk(tree))
    # Fixed point supports assignment aliases and iteration over produced values.
    for _ in range(len(nodes) + 1):
        before = dict(values)
        for node in nodes:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                kind = value(node.value)
                for target in targets:
                    if isinstance(target, ast.Name) and kind:
                        values[target.id] = kind
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
                kind = value(node.iter)
                if isinstance(node.target, ast.Name) and kind.endswith('-result'):
                    values[node.target.id] = kind
        if values == before:
            break
    for node in nodes:
        if isinstance(node, ast.Call):
            route = resolve(node.func)
            if route == 'nltk.pos_tag':
                return True
            if isinstance(node.func, ast.Attribute) and node.func.attr in GENERIC_NAMES:
                # A local definition also defeats an attribute route of that name.
                if node.func.attr not in shadow and not (isinstance(node.func.value, ast.Name) and node.func.value.id in shadow):
                    return True
            if isinstance(node.func, ast.Name) and route and route.rsplit('.', 1)[-1] in GENERIC_NAMES:
                return True
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'predict' and value(node.func.value) == 'flair-model':
                return True
        if isinstance(node, ast.Attribute):
            kind = value(node.value)
            if kind == 'spacy-result' and node.attr in ('pos_', 'tag_'):
                return True
            if kind == 'stanza-result' and node.attr in ('pos', 'upos', 'xpos'):
                return True
    return False


def audit(repo, ignore=(), verbose=None):
    root = Path(repo)
    if not root.is_dir():
        raise OSError('Repository is not a directory: ' + str(root))
    files = []
    excluded = {'.git', '__pycache__', '.venv', 'venv', 'env', '.env'}
    def walk(directory):
        for path in sorted(directory.iterdir()):
            rel = path.relative_to(root).as_posix()
            if path.is_symlink() or any(rel.startswith(p) for p in ignore):
                continue
            if path.is_dir():
                if path.name not in excluded and not (path / 'pyvenv.cfg').exists():
                    walk(path)
            else:
                files.append(path)
    walk(root)
    evidence = False
    docs = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        test = 'tests' in path.relative_to(root).parts[:-1]
        if path.suffix == '.py':
            present = has_evidence(path.read_text(encoding='utf-8'))
            evidence = evidence or present
            if verbose:
                verbose(rel + ': evidence=' + str(present))
        if rel == 'README.md' or (rel.startswith('docs/') and path.suffix == '.md') or (test and path.suffix in ('.py', '.md')):
            docs.append((rel, test, path.read_text(encoding='utf-8')))
    findings = []
    for rel, test, text in docs:
        denials, remaining = phrase_matches(prose(text), NOPOS)
        claims, _ = phrase_matches(remaining, POS)
        if verbose:
            verbose(rel + ': claims=' + str(len(claims)) + ', denials=' + str(len(denials)))
        for rule, phrases in [(POS_RULE, claims if not evidence else []), (NO_RULE, denials if evidence else [])]:
            for phrase in phrases:
                findings.append(dict(rule=rule, verdict='MEDIUM' if test else 'HIGH', message='Documentation and static source evidence disagree.', path=rel, phrase=phrase))
    return findings
