import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from part_of_speech_tagging_claim_audit.audit import audit, has_evidence, phrase_matches
from part_of_speech_tagging_claim_audit.phrases import POS, NOPOS, GENERIC_NAMES

POSITIVE = ['part-of-speech tagging', 'part of speech tagging', 'parts-of-speech tagging', 'part-of-speech tagger', 'part of speech tagger', 'pos tagging', 'pos tagger', 'pos tag', 'pos tags', 'tags parts of speech', 'tag parts of speech', 'part-of-speech tags', 'assigns part-of-speech tags', 'assign part-of-speech tags', 'grammatical tagging', 'morphosyntactic tagging', 'part-of-speech annotation', 'pos annotation', 'supports part-of-speech tagging', 'provides part-of-speech tagging', 'performs part-of-speech tagging', 'part-of-speech tagging support', 'part-of-speech tagging feature', 'part-of-speech tagging capability', 'part-of-speech tagging system', 'part-of-speech tagging model', 'pos tagging support', 'pos tagging feature', 'pos tagging capability', 'pos tagging system', 'pos tagging model', 'supports pos tagging', 'provides pos tagging', 'performs pos tagging', 'identifies parts of speech', 'identify parts of speech', 'detects parts of speech', 'labels parts of speech', 'label parts of speech', 'part-of-speech labels', 'grammatical categories', 'assigns grammatical tags', 'word class tagging', 'tags words with parts of speech', 'part-of-speech tag', 'part of speech tags', 'grammatical tag assignment', 'assigns pos tags', 'pos tag assignment', 'part-of-speech tag assignment']
NEGATIVE = ['does not tag parts of speech', 'do not tag parts of speech', "doesn't tag parts of speech", "don't tag parts of speech", 'no part-of-speech tagging', 'without part-of-speech tagging', 'never tags parts of speech', 'does not assign part-of-speech tags', 'no part-of-speech tagger', 'without a part-of-speech tagger', 'part-of-speech tagging not supported', 'part-of-speech tagging unsupported', 'no part-of-speech tagging support', 'does not perform part-of-speech tagging', 'never performs part-of-speech tagging', 'lacks part-of-speech tagging', 'missing part-of-speech tagging', 'part-of-speech tagging disabled', 'no pos tagging', 'without pos tagging', 'does not provide pos tagging', 'no pos tagger', 'pos tagging not supported', 'does not assign pos tags']
NAMES = ['tag_parts_of_speech', 'pos_tag_text', 'tag_pos', 'assign_pos_tags', 'get_pos_tags', 'pos_tagging', 'tag_words', 'grammatical_tag', 'annotate_pos', 'label_pos', 'predict_pos', 'pos_annotate', 'tag_with_pos', 'assign_grammatical_tags', 'part_of_speech_tag', 'pos_tag_sentence', 'tag_sentence_pos', 'get_part_of_speech_tags', 'identify_pos', 'detect_pos', 'extract_pos_tags', 'pos_label', 'tag_pos_words', 'annotate_parts_of_speech']

class AuditTests(unittest.TestCase):
    def fixture(self, files):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
        return root

    def cli(self, root, *args):
        return subprocess.run([sys.executable, '-m', 'part_of_speech_tagging_claim_audit', 'audit', str(root), *args], capture_output=True, text=True)

    def test_phrase_oracle(self):
        self.assertEqual(len(POSITIVE), 50)
        self.assertEqual(len(NEGATIVE), 24)
        self.assertEqual(POS, POSITIVE)
        self.assertEqual(NOPOS, NEGATIVE)

    def test_generic_oracle(self):
        self.assertEqual(len(NAMES), 24)
        self.assertEqual(GENERIC_NAMES, NAMES)

    def test_nltk(self):
        for code in ['import nltk\nnltk.pos_tag(words)', 'import nltk as n\nn.pos_tag(words)', 'from nltk import pos_tag\npos_tag(words)', 'from nltk import pos_tag as t\nt(words)']:
            with self.subTest(code=code):
                self.assertTrue(has_evidence(code))

    def test_spacy(self):
        for prefix, load in [('import spacy', 'spacy.load'), ('import spacy as s', 's.load'), ('from spacy import load as make', 'make')]:
            for attr in ['pos_', 'tag_']:
                for use in ['nlp=LOAD("en")\ndoc=nlp(text)\ndoc.ATTR', 'nlp=LOAD("en")\nnlp(text).ATTR', 'LOAD("en")(text).ATTR', 'nlp=LOAD("en")\nfor token in nlp(text):\n token.ATTR', 'nlp=LOAD("en")\n[token.ATTR for token in nlp(text)]']:
                    code = prefix + '\n' + use.replace('LOAD', load).replace('ATTR', attr)
                    with self.subTest(code=code):
                        self.assertTrue(has_evidence(code))

    def test_stanza(self):
        for prefix, make in [('import stanza', 'stanza.Pipeline'), ('import stanza as s', 's.Pipeline'), ('from stanza import Pipeline as P', 'P')]:
            for attr in ['pos', 'upos', 'xpos']:
                for use in ['p=MAKE(processors="tokenize,POS")\nd=p(text)\nd.ATTR', 'MAKE(processors="pos")(text).ATTR', 'p=MAKE(processors="pos")\nfor token in p(text):\n token.ATTR']:
                    code = prefix + '\n' + use.replace('MAKE', make).replace('ATTR', attr)
                    with self.subTest(code=code):
                        self.assertTrue(has_evidence(code))

    def test_flair(self):
        for prefix, model in [('import flair', 'flair.models.SequenceTagger'), ('import flair as f', 'f.models.SequenceTagger'), ('from flair.models import SequenceTagger', 'SequenceTagger'), ('from flair.models import SequenceTagger as T', 'T')]:
            for use in ['p=MODEL.load("POS-fast")\np.predict(text)', 'MODEL.load("pos").predict(text)']:
                code = prefix + '\n' + use.replace('MODEL', model)
                with self.subTest(code=code):
                    self.assertTrue(has_evidence(code))

    def test_all_generic_forms(self):
        for name in NAMES:
            for code in ['x.' + name + '(text)', 'from arbitrary import ' + name + '\n' + name + '(text)', 'from arbitrary import ' + name + ' as t\nt(text)']:
                with self.subTest(code=code):
                    self.assertTrue(has_evidence(code))

    def test_rejections(self):
        cases = [
            'import nltk; import spacy; import stanza; import flair',
            'from nltk import pos_tag',
            'import nltk\nnltk.ne_chunk(words)',
            'import spacy\nnlp=spacy.load("en")\nnlp(text).ents',
            'from transformers import pipeline\npipeline("ner")',
            'from flair.models import SequenceTagger\np=SequenceTagger.load("ner")\np.predict(text)',
            'x.extract_keywords(text)', 'model=KeyphraseModel()\nmodel(text)',
            'from transformers import pipeline\npipeline("sentiment-analysis")',
            'x.sentiment(text)',
            'from transformers import pipeline\npipeline("text-classification")',
            'classifier=Classifier()\nclassifier(text)',
            'from transformers import pipeline\npipeline("token-classification")',
            '"pos tagging"\n# nltk.pos_tag(words)',
            'from a import tag_parts_of_speech\ntag_parts_of_speech',
            'tag_parts_of_speech(text)',
            'import subprocess\nsubprocess.run(["pos-tagger", "file"])',
            'import spacy\nnlp=spacy.load("en")',
            'import stanza\np=stanza.Pipeline(processors="pos")',
            'from flair.models import SequenceTagger\np=SequenceTagger.load("pos")',
            'import stanza\np=stanza.Pipeline(processors="tokenize,ner")\np(text).upos',
            'import spacy\nnlp=spacy.load("en")\nother(text).pos_',
            'import spacy\nnlp=spacy.load("en")\nother.pos_',
            'import stanza\np=stanza.Pipeline(processors="pos")\nother(text).upos',
            'x.generate(text)',
        ]
        for code in cases:
            with self.subTest(code=code):
                self.assertFalse(has_evidence(code))

    def test_shadowing(self):
        cases = [
            'from a import tag_parts_of_speech\ndef tag_parts_of_speech(t): pass\ntag_parts_of_speech(text)',
            'import nltk\ndef nltk(): pass\nnltk.pos_tag(text)',
            'from nltk import pos_tag\ndef pos_tag(t): pass\npos_tag(text)',
            'from spacy import load\ndef load(t): pass\nload("en")(text).pos_',
            'from stanza import Pipeline\nclass Pipeline: pass\nPipeline(processors="pos")(text).upos',
            'from flair.models import SequenceTagger\nclass SequenceTagger: pass\np=SequenceTagger.load("pos")\np.predict(text)',
        ]
        for code in cases:
            with self.subTest(code=code):
                self.assertFalse(has_evidence(code))

    def test_split_modules(self):
        for first, second in [('import spacy\nnlp=spacy.load("en")', 'from a import nlp\nnlp(text).pos_'), ('import stanza\np=stanza.Pipeline(processors="pos")', 'from a import p\np(text).upos'), ('from flair.models import SequenceTagger\np=SequenceTagger.load("pos")', 'from a import p\np.predict(text)')]:
            root = self.fixture({'README.md': 'pos tagging', 'a.py': first, 'b.py': second})
            self.assertEqual(len(audit(root)), 1)

    def test_denial_span_removal(self):
        root = self.fixture({'README.md': 'does not tag parts of speech'})
        self.assertEqual(audit(root), [])

    def test_longest_first(self):
        found, _ = phrase_matches('part-of-speech tagging system', POS)
        self.assertEqual(found, ['part-of-speech tagging system'])

    def test_fenced_and_inline(self):
        self.assertEqual(audit(self.fixture({'README.md': '```text\npos tagging\n```'})), [])
        self.assertEqual(len(audit(self.fixture({'README.md': '`pos tagging`'}))), 1)

    def test_good(self):
        result = self.cli(self.fixture({'README.md': 'pos tagging', 'a.py': 'import nltk\nnltk.pos_tag(text)'}))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('PASS', result.stdout)

    def test_bad_json(self):
        result = self.cli(self.fixture({'README.md': 'pos tagging'}), '--json')
        self.assertEqual(result.returncode, 2)
        finding = json.loads(result.stdout)[0]
        self.assertEqual(finding['rule'], 'documented-part-of-speech-tagging-claim-false')
        self.assertEqual(finding['verdict'], 'HIGH')
        self.assertTrue(finding['message'])

    def test_test_only(self):
        root = self.fixture({'tests/fixture.py': '"pos tagging"'})
        result = self.cli(root, '--json')
        self.assertEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout)[0]['verdict'], 'MEDIUM')
        self.assertEqual(self.cli(root, '--fail-on', 'medium').returncode, 2)

    def test_denial(self):
        root = self.fixture({'README.md': 'no pos tagging', 'a.py': 'import nltk\nnltk.pos_tag(text)'})
        result = self.cli(root, '--json')
        self.assertEqual(result.returncode, 2)
        finding = json.loads(result.stdout)[0]
        self.assertEqual(finding['rule'], 'documented-no-part-of-speech-tagging-claim-false')
        self.assertEqual(finding['verdict'], 'HIGH')

    def test_no_claim(self):
        self.assertEqual(self.cli(self.fixture({'README.md': 'Plain utility.'})).returncode, 0)

    def test_version(self):
        result = subprocess.run([sys.executable, '-m', 'part_of_speech_tagging_claim_audit', '--version'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), '0.1.0')

    def test_ignore(self):
        root = self.fixture({'docs/a.md': 'pos tagging', 'README.md': 'Plain utility.'})
        self.assertEqual(self.cli(root, '--ignore', 'docs/', '--ignore', 'unused/').returncode, 0)
        root = self.fixture({'README.md': 'pos tagging', 'vendor/a.py': 'import nltk\nnltk.pos_tag(text)'})
        self.assertEqual(self.cli(root, '--ignore', 'vendor/').returncode, 2)

    def test_verbose(self):
        result = self.cli(self.fixture({'README.md': 'Plain utility.', 'a.py': 'x=1'}), '--verbose')
        self.assertEqual(result.returncode, 0)
        self.assertIn('Scan scope:', result.stderr)
        self.assertIn('a.py:', result.stderr)
        self.assertIn('README.md:', result.stderr)

    def test_errors(self):
        self.assertEqual(self.cli('/nonexistent/repository').returncode, 1)
        self.assertEqual(self.cli(self.fixture({'a.py': 'def broken('})).returncode, 1)
        self.assertEqual(self.cli(self.fixture({}), '--fail-on', 'low').returncode, 1)

    def test_excluded_directories(self):
        root = self.fixture({'README.md': 'pos tagging', '.git/a.py': 'import nltk\nnltk.pos_tag(t)', 'custom/pyvenv.cfg': '', 'custom/a.py': 'import nltk\nnltk.pos_tag(t)'})
        self.assertEqual(len(audit(root)), 1)

    def test_readme_self_audit(self):
        path = Path(__file__).resolve().parents[1] / 'README.md'
        if not path.exists():
            self.fail('Run the bundled tests from the source archive root')
        root = self.fixture({'README.md': path.read_text(encoding='utf-8')})
        self.assertEqual(audit(root), [])

if __name__ == '__main__':
    unittest.main()
