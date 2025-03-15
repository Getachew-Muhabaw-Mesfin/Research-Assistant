import os
import requests
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import google.auth

# Path to your OAuth 2.0 credentials JSON file
CLIENT_SECRET_FILE = './secrete.json'  # Replace with your file path
API_SCOPES = ['https://www.googleapis.com/auth/cloud-platform']

# Authenticate and get an access token
def authenticate():
    creds = None
    # Load credentials from the file if they exist
    if os.path.exists('token.json'):
        creds, _ = google.auth.load_credentials_from_file('token.json')
    # If no valid credentials are available, request new ones
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRET_FILE, API_SCOPES
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    return creds

# Get credentials and access token
credentials = authenticate()
access_token = credentials.token

# Define the request URL and headers
url = 'https://generativelanguage.googleapis.com/v1beta/models:predict'  # Make sure this endpoint is correct for listing models

headers = {
    'Authorization': f'Bearer {access_token}'
}

# Send request
response = requests.get(url, headers=headers)
models = response.json()

print(models)
