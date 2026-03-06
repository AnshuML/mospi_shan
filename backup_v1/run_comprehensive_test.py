import os
import sys
import pandas as pd
import json

# Add current directory to path so we can import app
sys.path.append('.')
import app

TEST_FILE = r'c:\Users\anshu\Desktop\eshankh\data\Semantic Search Test Report (1).xlsx'
OUTPUT_FILE = 'comprehensive_test_report.txt'

def parse_filters_string(s):
    """
    Parses a string like 'age group:Select All | state:India' into a dict
    """
    if not isinstance(s, str) or not s.strip():
        return {}
    res = {}
    parts = s.split('|')
    for p in parts:
        if ':' in p:
            k, v = p.split(':', 1)
            res[k.strip().lower()] = v.strip().lower()
    return res

def run_tests():
    xl = pd.ExcelFile(TEST_FILE)
    sheets = [s for s in xl.sheet_names if s != 'Summary']
    
    overall_summary = []
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as report:
        report.write("COMPREHENSIVE SEMANTIC SEARCH TEST REPORT\n")
        report.write("="*50 + "\n\n")
        
        for sheet in sheets:
            print(f"Testing {sheet}...")
            report.write(f"TESTING DATASET: {sheet}\n")
            report.write("-" * 30 + "\n")
            
            df = pd.read_excel(TEST_FILE, sheet_name=sheet)
            # Clean column names
            df.columns = [str(c).strip() for c in df.columns]
            
            sheet_correct = 0
            sheet_total = 0
            
            # Test only first 10 queries per sheet
            test_df = df.head(10)
            
            for i, row in test_df.iterrows():
                query = str(row.get('Prompts', ''))
                expected_ds = str(row.get('Expected Dataset', '')).strip()
                expected_ind = str(row.get('Expected Indicator', '')).strip()
                expected_filts_str = str(row.get('Expected Filters', ''))
                
                if not query or query == 'nan':
                    continue
                
                sheet_total += 1
                
                # 1. Rewrite Query
                rewritten = app.rewrite_query_with_llm(query)
                
                # 2. Search Indicators
                found_inds = app.search_indicators(rewritten)
                
                if not found_inds:
                    report.write(f"Row {i}: [FAIL] No indicators found for query: {query}\n")
                    continue
                
                # Get top result
                top_ind = found_inds[0]
                pred_ds = top_ind['parent']
                pred_ind = top_ind['name']
                
                # 3. Extract Filters for top indicator
                ind_code = top_ind['code']
                related_filters = [f for f in app.FILTERS if f["parent"] == ind_code]
                grouped = {}
                for f in related_filters:
                    grouped.setdefault(f["filter_name"], []).append(f)
                
                pred_filters = {}
                for fname, opts in grouped.items():
                    best_opt = app.select_best_filter_option(
                        query=rewritten,
                        raw_query=query,
                        filter_name=fname,
                        options=opts,
                        cross_encoder=app.cross_encoder
                    )
                    pred_filters[fname.lower()] = best_opt["option"].lower()
                
                # Validation
                ds_match = (pred_ds.lower() == expected_ds.lower())
                ind_match = (pred_ind.lower() == expected_ind.lower())
                
                # Filter matching is tricky, let's just log them
                exp_filters = parse_filters_string(expected_filts_str)
                filt_match = True
                for k, v in exp_filters.items():
                    if k in pred_filters:
                        if pred_filters[k] != v:
                            filt_match = False
                    # Note: Website might have filters not in products.json, we only check what we have
                
                if ds_match and ind_match and filt_match:
                    sheet_correct += 1
                    status = "PASS"
                else:
                    status = "FAIL"
                
                print(f"  Row {i}: {status}") 
                report.write(f"Row {i}: [{status}] Query: {query}\n")
                if status == "FAIL":
                    report.write(f"  Expected: DS={expected_ds}, IND={expected_ind}\n")
                    report.write(f"  Predicted: DS={pred_ds}, IND={pred_ind}\n")
                    report.write(f"  Exp Filts: {expected_filts_str}\n")
                    report.write(f"  Pred Filts: {pred_filters}\n")
                report.write("\n")
            
            acc = (sheet_correct / sheet_total * 100) if sheet_total > 0 else 0
            report.write(f"SHEET SUMMARY: {sheet_correct}/{sheet_total} ({acc:.2f}%)\n")
            report.write("\n\n")
            overall_summary.append((sheet, sheet_correct, sheet_total, acc))
            
        report.write("OVERALL SUMMARY\n")
        report.write("="*20 + "\n")
        total_q = 0
        total_c = 0
        for s, c, t, a in overall_summary:
            report.write(f"{s}: {c}/{t} ({a:.2f}%)\n")
            total_q += t
            total_c += c
        
        overall_acc = (total_c / total_q * 100) if total_q > 0 else 0
        report.write(f"\nTOTAL ACCURACY: {total_c}/{total_q} ({overall_acc:.2f}%)\n")

if __name__ == "__main__":
    run_tests()
