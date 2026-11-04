"""Command line interface."""
import argparse
import json
import sys
from . import __version__
from .audit import audit

class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(1, 'error: ' + message + '\n')

def main(argv=None):
    parser = Parser()
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True, parser_class=Parser)
    scan = sub.add_parser('audit')
    scan.add_argument('repo', nargs='?', default='.')
    scan.add_argument('--json', action='store_true')
    scan.add_argument('--verbose', action='store_true')
    scan.add_argument('--ignore', action='append', default=[])
    scan.add_argument('--fail-on', choices=['high', 'medium'], default='high')
    args = parser.parse_args(argv)
    try:
        log = (lambda message: print(message, file=sys.stderr)) if args.verbose else None
        if log:
            log('Scan scope: ' + args.repo)
        findings = audit(args.repo, args.ignore, log)
    except (OSError, UnicodeError, SyntaxError) as exc:
        print('error: ' + str(exc), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(findings, indent=2))
    else:
        for item in findings:
            print('{verdict} {rule} {path}: {message}'.format(**item))
        if not findings:
            print('PASS')
    return 2 if any(f['verdict'] == 'HIGH' or args.fail_on == 'medium' for f in findings) else 0

if __name__ == '__main__':
    sys.exit(main())
