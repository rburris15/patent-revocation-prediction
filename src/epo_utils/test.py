import json
import pandas as pd
def test_function():
    return "Module is working!"

def process_and_save_data(json_path):
    # Open the JSON file
    with open(json_path, 'r') as f:
        boa_pharma = json.load(f)

    def boa_to_df(boa):
        if len(boa['opponents_reps']) == 0:
            opponents = pd.DataFrame()
        else:
            opponents = pd.concat([
                pd.DataFrame({f"Opponent {i+1}":[v[0]], f"Representative {i+1}":[v[1]]}) 
                for i,v in enumerate(boa['opponents_reps'])
            ], axis=1)
        
        return pd.concat([
            pd.DataFrame.from_dict({
                "Decision date" : [boa['date']],
                "Case number" : [boa['case_number']],
                "Application number" : [boa['application_number']],
                "Publication number" : [boa['publication_number']],
                "IPC pharma" : [boa['IPC pharma']],
                "IPC biosimilar" : [boa['IPC biosimilar']],
                "IPCs" : [", ".join(boa['IPC'])],
                "Language" : [boa['lang']],
                "Title of Invention" : [boa['title_of_invention']],
                "Patent Proprietor" : [boa['patent_proprietor']],
                "Headword" : [boa['headword']],
                "Provisions" : [", ".join(boa['provisions'])],
                "Keywords" : [', '.join(boa['keywords'])],
                "Decisions cited" : [', '.join(boa['decisions_cited'])],
                "Summary" : ['\n\n'.join(boa['summary'])],
                "Decision reasons" : ['\n\n'.join(boa['decision_reasons'])],
                "Order" : [', '.join(boa['order'])],
                "Order status" : [boa['Order_status']],
                "Order status web" : [boa['Order_status_web']],
                "Order status manual" : [boa['Order_status_manual']],
                "Opponents" : [', '.join(boa['opponents'])]  
            }),
            opponents
        ], axis=1)

    boa_table = pd.concat([ boa_to_df(boa) for boa in boa_pharma], axis = 0)
    return boa_table