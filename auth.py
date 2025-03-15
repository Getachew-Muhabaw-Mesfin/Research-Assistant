import os
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import google.auth

# Path to your OAuth 2.0 credentials JSON file
CLIENT_SECRET_FILE = 'credentials.json'  # Ensure this file is in the correct path
API_SCOPES = ['https://www.googleapis.com/auth/cloud-platform']  # Adjust the scope if needed

# Set the redirect URI as used in Google Cloud Console
REDIRECT_URI = 'http://localhost:8080/oauth2callback'  # This must match the redirect URI in Google Cloud Console

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
            # OAuth flow with the specified redirect_uri
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRET_FILE, API_SCOPES
            )
            flow.redirect_uri = REDIRECT_URI  # Set the redirect URI here
            creds = flow.run_local_server(port=0)  # This will open the browser for authentication

        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    return creds

# Get credentials and access token
credentials = authenticate()
access_token = credentials.token

print(f"Access Token: {access_token}")
