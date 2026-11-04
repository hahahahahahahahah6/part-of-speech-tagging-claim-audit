# Documentation claim auditor

An offline checker compares documentation assertions about per-token word-class labels with source patterns. It uses only the standard library and makes no network requests, telemetry reports, or file writes.

## Usage

Requires Python 3.9 or newer.

```sh
python3 -m pip install part-of-speech-tagging-claim-audit
part-of-speech-tagging-claim-audit --version
part-of-speech-tagging-claim-audit audit .
python3 -m part_of_speech_tagging_claim_audit audit . --json
part-of-speech-tagging-claim-audit audit . --verbose --ignore vendor/ --ignore build/ --fail-on medium
```

The directory defaults to the current directory. Repeatable ignore prefixes apply to relative paths. Scanning reads the root readme, Markdown below the documentation directory, and fixture text below test directories. Source inspection reads Python files recursively, excluding version control, caches, virtual environments and symbolic links. A virtual environment configuration marker also excludes its directory.

```text
README.md
docs/**/*.md
tests/**/*.md
tests/**/*.py
```

Fenced code is excluded. Inline code counts as prose. Matching uses exact case-insensitive substrings, preserves hyphens, selects longer spans first, and removes denial spans before checking positive assertions.

Findings in shipped documentation have HIGH severity; test-only assertions have MEDIUM severity. Exit 0 means no finding meets the threshold; exit 2 means a finding meets it; exit 1 means a usage or input error. The default threshold is high. JSON output is a list whose findings include rule, verdict, message, path and phrase fields. Verdict contains severity. Verbose scope and file decisions go to standard error.

```text
documented-part-of-speech-tagging-claim-false
documented-no-part-of-speech-tagging-claim-false
```

## Evidence

Imports resolve through aliases within each module. Local function and class definitions shadow imported names. Paired routes track construction, invocation and result access within one file.

```text
nltk.pos_tag(...)
spacy.load(...) + loaded-object call + result.pos_ or result.tag_
stanza.Pipeline(processors="tokenize,pos") + instance call + result.pos/upos/xpos
flair.models.SequenceTagger.load("pos") + instance.predict(...)
24 exact generic callable names
```

Token iteration over produced values is supported. Imports alone, unrelated task calls, strings, comments, uncalled references and incomplete pairs do not establish evidence.

## Differentiation

The neighboring tools inspect different outputs:

```text
named-entity-recognition-claim-audit: entity spans vs per-token word-class labels
sentiment-claim-audit: polarity score vs per-token word-class labels
classification-claim-audit: single label vs per-token word-class labels
```

An entity span locates a named referent. A polarity score describes tone. A single document label categorizes the input. This checker examines assertions about the word-class label of each token.

## Limitations

Static pattern matching is not semantic judgement and cannot prove runtime behavior or label quality. Exact phrases miss paraphrases. Generic method names may yield false positives. Dynamic dispatch, wrappers and cross-module flow may be missed. Files are parsed without execution; unreadable or malformed scanned files cause an input error.

## Development

```sh
python3 -m unittest discover
python3 -m build
```

The source archive includes the tests. Licensed under MIT.
