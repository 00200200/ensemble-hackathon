import json
from nltk.translate.chrf_score import corpus_chrf
import sys
import nltk

def evaluate_chrf(predictions_file: str, ground_truth_file: str, source_file: str):
    """
    Evaluates the chrF score of the predicted context against the ground truth middle.
    Since ground_truths don't have IDs, we zip them with the source file to align IDs.
    """
    
    # Load ground truths with IDs from source
    truths = {}
    with open(source_file, 'r', encoding='utf-8') as fs, \
         open(ground_truth_file, 'r', encoding='utf-8') as fg:
        for line_s, line_g in zip(fs, fg):
            data_s = json.loads(line_s)
            data_g = json.loads(line_g)
            truths[data_s['id']] = data_g['middle']
            
    # Load predictions
    preds = {}
    with open(predictions_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            preds[data['id']] = data['context']
            
    # Align
    refs = []
    hyps = []
    
    missing = 0
    for qid, middle_text in truths.items():
        if qid in preds:
            refs.append([middle_text])  # chrF expects a list of references for each hypothesis
            hyps.append(preds[qid])
        else:
            missing += 1
            
    if missing > 0:
        print(f"Warning: {missing} predictions missing from your file.")
        
    print(f"Evaluating {len(hyps)} predictions...")
    
    # Calculate corpus chrF score
    score = corpus_chrf(refs, hyps)
    print(f"Approximated Context-to-Target chrF Score: {score:.4f}")
    
    
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python evaluate_local.py <predictions.jsonl> <ground_truths.jsonl> <source_data.jsonl>")
        sys.exit(1)
        
