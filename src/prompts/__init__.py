from prompts.base import (
    ExampleTemplate,
    GenerationTemplate,
    ClassificationTemplate,
    Accuracy, Rouge, BreakLFEM,
    # GenerationExampleTemplate,
    # ClassificationExampleTemplate,
    # ContextualizedGenerationExampleTemplate,
    # ContextualizedClassificationExampleTemplate,
    # SemparseExampleTemplate
)
from prompts.few_shot import FewShotPromptTemplate2
from prompts.mwp import GSM8KExampleTemplate, AquaExampleTemplate