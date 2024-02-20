import attr
import typing
import evaluate
from tools.param_impl import DictDataClass

class Metric:
    def __call__(self, pred, **kwargs):
        raise NotImplementedError
class Accuracy(Metric):
    def __call__(self, pred, _target, **kwargs):
        return dict(accuracy=(pred == _target) * 100)

class BreakLFEM():
    def __init__(self, qdecomp_path='third_party/qdecomp_with_dependency_graphs'):
        import sys
        if qdecomp_path not in sys.path:
            sys.path.append(qdecomp_path)
        from dependencies_graph.evaluation.logical_form_matcher import LogicalFromStructuralMatcher
        from dependencies_graph.evaluation.qdmr_to_logical_form_tokens import \
            QDMRToQDMRStepTokensConverter
        from evaluation.normal_form.normalized_graph_matcher import \
            NormalizedGraphMatchScorer
        from scripts.eval.evaluate_predictions import format_qdmr
        self.converter = QDMRToQDMRStepTokensConverter()
        self.matcher = LogicalFromStructuralMatcher()
        self.scorer = NormalizedGraphMatchScorer()
        self.format_qdmr = format_qdmr
        sys.path.pop()
    def __call__(self, pred, _target, **kwargs):
        lfem = False
        try:
            generated = pred
            question = kwargs['question_text']
            decomposition = _target
            index = kwargs['question_id']

            gold = self.format_qdmr(decomposition)
            pred = self.format_qdmr(generated)
            decomp_lf = self.converter.convert(question_id=str(index), question_text=question,
                                                decomposition=pred.to_break_standard_string())
            gold_lf = self.converter.convert(question_id=str(index), question_text=question,
                                                decomposition=gold.to_break_standard_string())
            lfem = self.matcher.is_match(question_id=str(index), question_text=question, graph1=decomp_lf,
                                        graph2=gold_lf)
        except Exception as e:
            print(e)
        return dict(lfem=lfem * 100)

class Rouge(Metric):
    def __init__(self):
        import evaluate
        self.rouge = evaluate.load('rouge')

    def __call__(self, pred, _target, **kwargs):
        rouge_results = self.rouge.compute(
            predictions=[pred], references=[_target], use_stemmer=True
        )
        return {k: v * 100 for k, v in rouge_results.items()}

class GSM8KAccuracy(Metric):
    def extract_answer(self, completion):
        import re
        ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
        INVALID_ANS = "[invalid]"
        match = ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            # special case for turbo
            TURBO_ANS_RE = re.compile(r"Answer: \\boxed{(\-?[0-9\.\,]+)}")
            match = TURBO_ANS_RE.search(completion)
            if match:
                match_str = match.group(1).strip()
                match_str = match_str.replace(",", "")
                return match_str
            else:
                return INVALID_ANS

    def __call__(self, pred, _target, **kwargs):
        is_correct = self.extract_answer(pred) == self.extract_answer(_target)
        return dict(accuracy=is_correct * 100)

class TabMWPAccuracy(Metric):

    def __init__(self, option_inds: list[str]) -> None:
        self.option_inds = option_inds

    def __call__(self, pred, _target, **kwargs):
        from prompts.tabmwp import extract_prediction, normalize_answer
        prediction = extract_prediction(pred, kwargs['choices'], self.option_inds)
        prediction = normalize_answer(prediction, kwargs['unit'])
        target = extract_prediction(_target, kwargs['choices'], self.option_inds)
        target = normalize_answer(target, kwargs['unit'])
        is_correct = prediction.lower() == target.lower()
        return dict(accuracy=is_correct * 100)

@attr.s(auto_attribs=True)
class ExampleTemplate(DictDataClass):
    templates: dict[str, typing.Callable] | dict[str, str] | str
    get_target: str | typing.Callable = 'target'
    metrics: list[Metric] = attr.ib(factory=lambda: [Accuracy()])
    result_metric: str = 'accuracy'

    def format(self, test=False, **kwargs):
        raise NotImplementedError

    def parse_output(self, lm_output: str, **kwargs) -> str:
        return lm_output.strip()

    def check_output(self, pred, _target, **kwargs):
        results = {}
        for metric in self.metrics:
            results |= metric(pred, _target=_target, **kwargs)
        results['result'] = results[self.result_metric]
        return results

@attr.s(auto_attribs=True)
class GenerationTemplate(ExampleTemplate):
    get_target: str | typing.Callable = 'target'

    def __attrs_post_init__(self):
        if isinstance(self.templates, str):
            train_template = self.templates
            # assert train_template.endswith(r'{_target}')
            if train_template.endswith(r'{_target}'):
                test_template = train_template[:-len(r'{_target}')]
            else:
                test_template = train_template
            self.templates = dict(train=train_template, test=test_template)
        assert isinstance(self.templates, dict)
        self.templates = {
            k: t.format if isinstance(t, str) else t
            for k, t in self.templates.items()
        }
        if isinstance(self.get_target, str):
            target_variable = self.get_target   # can't pass self.get_target directly to lambda function
            self.get_target = lambda **kwargs: kwargs[target_variable]

    def format(self, test=False, **kwargs):
        kwargs['_target'] = self.get_target(**kwargs)
        if test: return self.templates['test'](**kwargs)
        else: return self.templates['train'](**kwargs)

@attr.s(auto_attribs=True)
class ClassificationTemplate(ExampleTemplate):
    choices: list[str] | dict[str, str] | typing.Callable = None
    get_target: typing.Callable = lambda choices, **kwargs: choices[kwargs['label']]

    def __attrs_post_init__(self):
        if isinstance(self.templates, str):
            train_template = self.templates
            # assert train_template.endswith(r'{_target}')
            # test_template = train_template[:-len(r'{_target}')]
            if train_template.endswith(r'{_target}'):
                test_template = train_template[:-len(r'{_target}')]
            else:
                test_template = train_template
            self.templates = dict(train=train_template, test=test_template)
        assert isinstance(self.templates, dict)
        self.templates = {
            k: t.format if isinstance(t, str) else t
            for k, t in self.templates.items()
        }
        # get_target = self.get_target
        # self.get_target = lambda **kwargs: get_target(self.get_choices(**kwargs), **kwargs)
    def get_choices(self, **kwargs):
        if isinstance(self.choices, (list, dict)):
            return self.choices
        else:
            return self.choices(**kwargs)
    def format(self, test=False, **kwargs):
        kwargs['_target'] = self.get_target(self.get_choices(**kwargs), **kwargs)

        if test: return self.templates['test'](**kwargs)
        else: return self.templates['train'](**kwargs)

    def check_output(self, pred, **kwargs):
        kwargs['_target'] = self.get_target(self.get_choices(**kwargs), **kwargs)
        return super().check_output(pred, **kwargs)
