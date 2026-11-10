import sacrebleu
from sacrebleu.metrics import BLEU, CHRF


def corpus_metrics(hypotheses, references):
    if not hypotheses or len(hypotheses) != len(references):
        raise ValueError("Hypotheses and references must be nonempty and aligned")
    hypotheses = [" ".join(s.split()) for s in hypotheses]
    references = [" ".join(s.split()) for s in references]
    metric = BLEU(
        tokenize="none", lowercase=False, smooth_method="exp", effective_order=False, force=True
    )
    score = metric.corpus_score(hypotheses, [references])
    chrf = CHRF(beta=2, word_order=0)
    chrf_score = chrf.corpus_score(hypotheses, [references])
    return {
        "bleu": score.score,
        "bleu_signature": str(metric.get_signature()),
        "chrf2": chrf_score.score,
        "chrf_signature": str(chrf.get_signature()),
        "sacrebleu_version": sacrebleu.__version__,
        "sentences": len(hypotheses),
        "hypothesis_tokens": score.sys_len,
        "reference_tokens": score.ref_len,
        "brevity_penalty": score.bp,
        "text_format": "moses_tokenized",
    }
