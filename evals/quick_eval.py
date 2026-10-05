"""
Quick Evaluation Suite for Enterprise Agentic RAG LLM System.
Runs:
1. Health Check
2. Safety & Guardrails Benchmark (6 tests: jailbreak, prompt injection, off-topic, legit)
3. Live Agentic RAG Benchmark (3 golden multi-domain queries)
4. Tool Selection Accuracy (Jaccard)
5. RAGAS Judge Evaluation (Faithfulness, Answer Relevancy, Answer Correctness)
"""

import os
import sys
import time
import json
import asyncio
from dotenv import load_dotenv

load_dotenv()

# Add root dir to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from evals.guardrails_eval import run_guardrails_eval, compute_guardrails_metrics
from evals.pipeline import detect_tool, load_golden_dataset
from evals.metrics import _build_judge, Faithfulness, AnswerRelevancy, AnswerCorrectness

API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/") + "/query"
BASE_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")


def check_backend_health():
    print(f"📡 Checking backend at {BASE_URL}...")
    try:
        r = requests.get(BASE_URL, timeout=10)
        if r.status_code == 200:
            print("  ✅ Backend is online and healthy.\n")
            return True
        else:
            print(f"  ❌ Backend returned status code {r.status_code}\n")
            return False
    except Exception as e:
        print(f"  ❌ Failed to connect to backend: {e}\n")
        return False


def run_quick_guardrails(golden_dataset):
    print("=" * 70)
    print("🛡️  PHASE 1: GUARDRAILS SAFETY EVALUATION (6 TEST CASES)")
    print("=" * 70)
    samples = golden_dataset.get("guardrails_samples", [])
    
    t0 = time.time()
    results = run_guardrails_eval(samples)
    duration = time.time() - t0
    
    metrics = compute_guardrails_metrics(results)
    
    print(f"\nCompleted in {duration:.1f}s")
    print(f"Total Tests: {metrics['total']} | Correct: {metrics['correct']} | Accuracy: {metrics['accuracy'] * 100:.1f}%")
    print(f"Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | TP: {metrics['tp']} | TN: {metrics['tn']} | FP: {metrics['fp']} | FN: {metrics['fn']}\n")
    
    for r in results:
        status_symbol = "✅" if r["result"] in ("TP", "TN") else "❌"
        exp = "Block" if r["expected_blocked"] else "Pass"
        act = "Blocked" if r["actual_blocked"] else "Passed"
        print(f"  {status_symbol} [{r['id']}] Expected: {exp:5} | Actual: {act:7} | Result: {r['result']:2} | Input: {r['input'][:60]}...")
    print()
    return results, metrics


def run_quick_rag_pipeline(golden_dataset, sample_ids=(1, 4, 14)):
    print("=" * 70)
    print(f"🚀  PHASE 2: LIVE AGENTIC RAG PIPELINE (SAMPLE IDS: {sample_ids})")
    print("=" * 70)
    
    all_samples = {s["id"]: s for s in golden_dataset.get("rag_samples", [])}
    chosen_samples = [all_samples[sid] for sid in sample_ids if sid in all_samples]
    
    enriched = []
    
    for idx, s in enumerate(chosen_samples, 1):
        q = s["question"]
        print(f"\n[{idx}/{len(chosen_samples)}] Querying: \"{q}\"")
        start_t = time.time()
        
        try:
            resp = requests.post(
                API_URL,
                json={"q": q, "thread_id": f"quick_eval_{s['id']}"},
                timeout=120
            )
            resp.raise_for_status()
            data = resp.json()
            latency = time.time() - start_t
            
            raw_answer = data.get("answer") or ""
            thought_process = data.get("thought_process") or []
            sources = data.get("sources") or []
            detected = detect_tool(thought_process)
            
            # Tool correctness (Jaccard)
            called_set = {detected}
            expected_set = set(s.get("expected_tools", []))
            union = len(called_set | expected_set)
            tool_score = len(called_set & expected_set) / union if union > 0 else 0.0
            
            item = {
                "id": s["id"],
                "domain": s.get("domain", ""),
                "question": q,
                "reference": s.get("reference", ""),
                "expected_tools": s.get("expected_tools", []),
                "actual_response": raw_answer,
                "actual_contexts": sources[:3],
                "actual_tools_called": [detected],
                "thought_process": thought_process,
                "latency_sec": round(latency, 2),
                "tool_score": tool_score,
                "status": data.get("status", "completed")
            }
            enriched.append(item)
            
            print(f"  ⚡ Latency: {latency:.2f}s | Context chunks: {len(sources)} | Tool: {detected} (Match: {tool_score:.2f})")
            print(f"  📝 Answer preview: {raw_answer[:140]}...")
            
        except Exception as e:
            print(f"  ❌ Query failed: {e}")
            item = {
                "id": s["id"],
                "domain": s.get("domain", ""),
                "question": q,
                "reference": s.get("reference", ""),
                "expected_tools": s.get("expected_tools", []),
                "actual_response": "",
                "actual_contexts": s.get("relevant_contexts", []),
                "actual_tools_called": ["unknown"],
                "thought_process": [str(e)],
                "latency_sec": 0,
                "tool_score": 0.0,
                "status": "error"
            }
            enriched.append(item)
            
        # Small breath between queries to respect Groq rate limits
        if idx < len(chosen_samples):
            time.sleep(3)
            
    return enriched


async def evaluate_ragas_metrics(enriched_samples):
    print("\n" + "=" * 70)
    print("📊  PHASE 3: RAGAS LLM-AS-A-JUDGE METRICS")
    print("=" * 70)
    print("Initializing Judge LLM (llama-3.1-8b-instant) and MiniLM Embeddings...")
    
    judge_llm, ragas_embeddings = _build_judge()
    print("  ✅ Judge and Embeddings initialized.")
    
    # Filter samples that received responses
    valid_samples = [s for s in enriched_samples if s.get("actual_response", "").strip()]
    if not valid_samples:
        print("  ⚠️ No responses to evaluate.")
        return {}
        
    scores_by_sample = {s["id"]: {} for s in valid_samples}
    
    # 1. Faithfulness
    print("\n  🔍 1/3 Evaluating Faithfulness (Groundedness in context)...")
    faith_metric = Faithfulness(llm=judge_llm)
    faith_inputs = [
        {
            "user_input": s["question"],
            "response": s["actual_response"][:500],
            "retrieved_contexts": [c[:400] for c in (s["actual_contexts"] or [""])[:2]],
        }
        for s in valid_samples
    ]
    try:
        faith_scores = await faith_metric.abatch_score(faith_inputs)
        for s, score_obj in zip(valid_samples, faith_scores):
            val = round(float(score_obj.value), 3) if hasattr(score_obj, "value") else 0.0
            scores_by_sample[s["id"]]["faithfulness"] = val
            print(f"     Sample #{s['id']}: {val:.3f}")
    except Exception as e:
        print(f"     ❌ Faithfulness scoring error: {e}")
        for s in valid_samples:
            scores_by_sample[s["id"]]["faithfulness"] = None

    # Cooldown for Groq TPM
    print("     ⏳ 10s cooldown buffer...")
    await asyncio.sleep(10)

    # 2. Answer Relevancy
    print("\n  🎯 2/3 Evaluating Answer Relevancy (Directness to question)...")
    ans_rel_metric = AnswerRelevancy(llm=judge_llm, embeddings=ragas_embeddings)
    ans_rel_inputs = [
        {
            "user_input": s["question"],
            "response": s["actual_response"][:500],
        }
        for s in valid_samples
    ]
    try:
        rel_scores = await ans_rel_metric.abatch_score(ans_rel_inputs)
        for s, score_obj in zip(valid_samples, rel_scores):
            val = round(float(score_obj.value), 3) if hasattr(score_obj, "value") else 0.0
            scores_by_sample[s["id"]]["answer_relevancy"] = val
            print(f"     Sample #{s['id']}: {val:.3f}")
    except Exception as e:
        print(f"     ❌ Answer Relevancy scoring error: {e}")
        for s in valid_samples:
            scores_by_sample[s["id"]]["answer_relevancy"] = None

    # Cooldown for Groq TPM
    print("     ⏳ 10s cooldown buffer...")
    await asyncio.sleep(10)

    # 3. Answer Correctness
    print("\n  ✅ 3/3 Evaluating Answer Correctness (Factual match with ground truth)...")
    ans_corr_metric = AnswerCorrectness(llm=judge_llm, embeddings=ragas_embeddings)
    ans_corr_inputs = [
        {
            "user_input": s["question"],
            "response": s["actual_response"][:500],
            "reference": s["reference"][:500],
        }
        for s in valid_samples
    ]
    try:
        corr_scores = await ans_corr_metric.abatch_score(ans_corr_inputs)
        for s, score_obj in zip(valid_samples, corr_scores):
            val = round(float(score_obj.value), 3) if hasattr(score_obj, "value") else 0.0
            scores_by_sample[s["id"]]["answer_correctness"] = val
            print(f"     Sample #{s['id']}: {val:.3f}")
    except Exception as e:
        print(f"     ❌ Answer Correctness scoring error: {e}")
        for s in valid_samples:
            scores_by_sample[s["id"]]["answer_correctness"] = None

    return scores_by_sample


def main():
    print("=" * 70)
    print("🚀 ENTERPRISE AGENTIC RAG — QUICK MODEL EVALUATION SUITE")
    print("=" * 70)
    
    if not check_backend_health():
        sys.exit(1)
        
    golden = load_golden_dataset()
    
    # 1. Guardrails
    guard_results, guard_metrics = run_quick_guardrails(golden)
    
    # 2. Live Agentic RAG
    rag_results = run_quick_rag_pipeline(golden, sample_ids=[1, 4, 14])
    
    # 3. RAGAS Metrics
    ragas_scores = asyncio.run(evaluate_ragas_metrics(rag_results))
    
    # Combine results
    final_output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "guardrails_metrics": guard_metrics,
        "guardrails_cases": guard_results,
        "rag_evaluation": []
    }
    
    for r in rag_results:
        sid = r["id"]
        sample_ragas = ragas_scores.get(sid, {})
        entry = {
            "id": sid,
            "domain": r["domain"],
            "question": r["question"],
            "reference": r["reference"],
            "actual_response": r["actual_response"],
            "latency_sec": r["latency_sec"],
            "tool_called": r["actual_tools_called"][0],
            "tool_correctness": r["tool_score"],
            "faithfulness": sample_ragas.get("faithfulness"),
            "answer_relevancy": sample_ragas.get("answer_relevancy"),
            "answer_correctness": sample_ragas.get("answer_correctness"),
        }
        final_output["rag_evaluation"].append(entry)
        
    out_path = os.path.join(os.path.dirname(__file__), "quick_eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
        
    print("\n" + "=" * 70)
    print(f"🎉 EVALUATION COMPLETE — RESULTS SAVED TO {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
