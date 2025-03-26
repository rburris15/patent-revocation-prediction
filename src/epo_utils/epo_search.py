import requests
import pandas as pd

def get_is_citations(application_number):
    """
    Fetch the number of International Search citations for a given application number from the EPO OPS API.
    """
    base_url = "https://ops.epo.org/rest-services/published-data/publication/epodoc/{}/citations".format(application_number)
    headers = {
        "Accept": "application/json",
        "User-Agent": "YourAppNameHere",
        "Authorization": "Bearer YOUR_ACCESS_TOKEN"  # Replace with actual token
    }
    
    response = requests.get(base_url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        # Extract number of IS citations from response (adjust based on actual API structure)
        is_citations = len(data.get("citation", []))  # Placeholder, adjust based on API response format
        return is_citations
    else:
        return None  # Return None for failed requests

def add_is_citations_column(df, application_column):
    """
    Add a new column "IS Citations" to the dataframe with the count of International Search citations.
    """
    df["IS Citations"] = df[application_column].apply(get_is_citations)
    return df

# Example usage:
# df = pd.DataFrame({"Application Number": ["EP1234567", "EP7654321"]})
# df = add_is_citations_column(df, "Application Number")
# print(df)