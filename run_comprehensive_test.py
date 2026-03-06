import os
import sys
import pandas as pd
import json
import time

# Add current directory to path so we can import app
sys.path.append('.')
import app as app

TEST_FILE = r'c:\Users\anshu\Desktop\eshankh\data\Semantic Search Test Report (1).xlsx'
OUTPUT_FILE = 'enterprise_test_report.txt'

def parse_filters_string(s):
    if not isinstance(s, str) or not s.strip():
        return {}
    res = {}
    parts = s.split('|')
    for p in parts:
        if ':' in p:
            k, v = p.split(':', 1)
            res[k.strip().lower()] = v.strip().lower()
    return res

def run_golden_queries():
    print("\n" + "="*50)
    print("RUNNING GOLDEN QUERIES (Secretary Meeting Safety)")
    print("="*50)
    
    golden_tests = [
        {"query": "What was India's GDP in 2023-24?", "expected_ds": "NAS", "expected_ind": "Back"},
        {"query": "Total factories in Gujarat for 2022-23", "expected_ds": "ASI", "expected_ind": "2008"},
        {"query": "Wholesale price of Potato in January 2024", "expected_ds": "WPI", "expected_ind": "Primary articles"},
        {"query": "IIP for Mining in June 1998", "expected_ds": "IIP", "expected_ind": "2004-05"}
    ]
    
    passed = 0
    for test in golden_tests:
        print(f"\nQuery: {test['query']}")
        clean_q = app.rewrite_query_with_llm(test['query'])
        results = app.enterprise_hybrid_search(clean_q)
        
        if not results:
            print("  [FAIL] No results")
            continue
            
        best = results[0]
        ds_match = best['parent'] == test['expected_ds']
        ind_match = best['name'] == test['expected_ind']
        
        status = "PASS" if ds_match and ind_match else "FAIL"
        print(f"  Predicted: DS={best['parent']}, IND={best['name']}")
        print(f"  Confidence: {best.get('final_score', 0):.2f}")
        print(f"  Status: {status}")
        if status == "PASS": passed += 1
        
    print(f"\nGOLDEN QUERIES PASSED: {passed}/{len(golden_tests)}")
    return passed == len(golden_tests)

def run_full_verification():
    xl = pd.ExcelFile(TEST_FILE)
    sheets = [s for s in xl.sheet_names if s != 'Summary']
    
    overall_summary = []
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as report:
        report.write("ENTERPRISE HYBRID SEARCH TEST REPORT (MoSPI V2)\n")
        report.write("="*50 + "\n\n")
        
        for sheet in sheets:
            print(f"\nTesting Dataset: {sheet}")
            report.write(f"TESTING DATASET: {sheet}\n")
            report.write("-" * 30 + "\n")
            
            df = pd.read_excel(TEST_FILE, sheet_name=sheet)
            df.columns = [str(c).strip() for c in df.columns]
            
            sheet_correct = 0
            sheet_total = 0
            
            # Test 10 samples per product
            test_df = df.head(10)
            
            for i, row in test_df.iterrows():
                query = str(row.get('Prompts', ''))
                expected_ds = str(row.get('Expected Dataset', '')).strip()
                expected_ind = str(row.get('Expected Indicator', '')).strip()
                expected_filts_str = str(row.get('Expected Filters', ''))
                
                if not query or query == 'nan': continue
                
                sheet_total += 1
                
                # Full Prediction Cycle
                clean_q = app.rewrite_query_with_llm(query)
                found = app.enterprise_hybrid_search(clean_q, raw_query=query)
                
                if not found:
                    report.write(f"Row {i}: [FAIL] No indicators found\n\n")
                    continue
                    
                best = found[0]
                pred_ds = best['parent']
                pred_ind = best['name']
                pred_filters = app.resolve_filters(clean_q, query, best["code"])
                
                # Validation Logic
                ds_match = (pred_ds.lower() == expected_ds.lower())
                # Handle cases where expected ind is slightly different string but same meaning
                ind_match = (pred_ind.lower() in expected_ind.lower() or expected_ind.lower() in pred_ind.lower())
                
                if ds_match and ind_match:
                    sheet_correct += 1
                    status = "PASS"
                else:
                    status = "FAIL"
                
                print(f"  Row {i}: {status} (Score: {best.get('final_score', 0):.2f})")
                report.write(f"Row {i}: [{status}] Query: {query}\n")
                if status == "FAIL":
                    report.write(f"  Exp: DS={expected_ds}, IND={expected_ind}\n")
                    report.write(f"  Pred: DS={pred_ds}, IND={pred_ind}\n")
                report.write("\n")
                
            acc = (sheet_correct / sheet_total * 100) if sheet_total > 0 else 0
            report.write(f"SHEET SUMMARY: {sheet_correct}/{sheet_total} ({acc:.2f}%)\n\n")
            overall_summary.append((sheet, sheet_correct, sheet_total, acc))
            
        report.write("OVERALL SUMMARY (V2)\n")
        report.write("="*20 + "\n")
        for s, c, t, a in overall_summary:
            report.write(f"{s}: {c}/{t} ({a:.2f}%)\n")

if __name__ == "__main__":
    if run_golden_queries():
        print("\nGolden Queries look solid. Starting full verification...")
    else:
        print("\nGolden Queries failed. Reviewing logic...")
        
    run_full_verification()
