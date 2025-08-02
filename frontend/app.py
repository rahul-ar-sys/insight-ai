import streamlit as st
import requests
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configure page - MUST be first Streamlit command
st.set_page_config(
    page_title="Insight AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for wider sidebar and better styling
st.markdown("""
<style>
    /* Make sidebar wider */
    .css-1d391kg {
        width: 400px !important;
    }
    
    /* Adjust main content to account for wider sidebar */
    .css-18e3th9 {
        padding-left: 420px !important;
    }
    
    /* Sidebar background and styling */
    .css-1lcbmhc {
        background-color: #f8f9fa;
        border-right: 2px solid #e9ecef;
    }
    
    /* Sidebar toggle button styling */
    .sidebar-toggle {
        position: fixed;
        top: 10px;
        left: 10px;
        z-index: 1000;
        background: #007bff;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 8px 12px;
        cursor: pointer;
        font-size: 16px;
    }
    
    .sidebar-toggle:hover {
        background: #0056b3;
    }
    
    /* Responsive design for smaller screens */
    @media (max-width: 768px) {
        .css-1d391kg {
            width: 300px !important;
        }
        .css-18e3th9 {
            padding-left: 320px !important;
        }
    }
    
    /* Document container styling */
    .document-container {
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        background: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Processing status indicators */
    .status-completed { color: #28a745; }
    .status-processing { color: #007bff; }
    .status-error { color: #dc3545; }
    .status-pending { color: #6c757d; }
</style>
""", unsafe_allow_html=True)

# Constants
API_BASE_URL = "http://127.0.0.1:8000"  # Update with your API base URL

# Initialize session state immediately after page config
def init_session_state():
    """Initialize all session state variables"""
    # Core client
    if "client" not in st.session_state:
        st.session_state.client = None  # Will be set after class definition
    
    # Authentication state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user" not in st.session_state:
        st.session_state.user = None
    
    # Firebase state
    if "firebase_auth_pending" not in st.session_state:
        st.session_state.firebase_auth_pending = False
    if "firebase_user" not in st.session_state:
        st.session_state.firebase_user = None
    if "firebase_token" not in st.session_state:
        st.session_state.firebase_token = None
    
    # App state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_workspace" not in st.session_state:
        st.session_state.current_workspace = None
    if "workspaces" not in st.session_state:
        st.session_state.workspaces = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "show_knowledge_graph" not in st.session_state:
        st.session_state.show_knowledge_graph = False

# Initialize immediately
init_session_state()

# Firebase Configuration
# Note: In production, these should come from environment variables
FIREBASE_CONFIG = {
    "apiKey": "your-firebase-api-key",  # Replace with your actual Firebase API key
    "authDomain": "your-project.firebaseapp.com",  # Replace with your domain
    "projectId": "your-project-id",  # Replace with your project ID
    "storageBucket": "your-project.appspot.com",
    "messagingSenderId": "your-sender-id",
    "appId": "your-app-id"
}

# For development, disable Firebase if not properly configured
FIREBASE_ENABLED = all(
    "your-" not in str(value) for value in FIREBASE_CONFIG.values()
)

# Global CSS Styles
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.main-header h1 {
    margin: 0;
    font-size: 3rem;
    font-weight: 700;
}

.main-header p {
    margin: 0.5rem 0 0 0;
    font-size: 1.2rem;
    opacity: 0.9;
}

.chat-message {
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 10px;
    border-left: 4px solid #667eea;
    background-color: #f8f9ff;
}

.user-message {
    background-color: #e8f4fd;
    border-left-color: #1f77b4;
}

.assistant-message {
    background-color: #f0f8f0;
    border-left-color: #2ca02c;
}

.auth-container {
    max-width: 400px;
    margin: 2rem auto;
    padding: 2rem;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    background-color: white;
}

.firebase-auth {
    margin: 1rem 0;
}

.google-signin-btn {
    background-color: #4285f4;
    color: white;
    padding: 12px 20px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    width: 100%;
    font-size: 16px;
    margin: 10px 0;
}

.google-signin-btn:hover {
    background-color: #357ae8;
}

.auth-divider {
    text-align: center;
    margin: 20px 0;
    position: relative;
}

.auth-divider::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 0;
    right: 0;
    height: 1px;
    background: #ddd;
}

.auth-divider span {
    background: white;
    padding: 0 15px;
    color: #666;
}

.developer-login {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 5px;
    padding: 1rem;
    margin-top: 1rem;
}

.success-message {
    background-color: #d4edda;
    color: #155724;
    padding: 0.75rem;
    border-radius: 0.25rem;
    border: 1px solid #c3e6cb;
    margin: 1rem 0;
}

.error-message {
    background-color: #f8d7da;
    color: #721c24;
    padding: 0.75rem;
    border-radius: 0.25rem;
    border: 1px solid #f5c6cb;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)


class InsightAIClient:
    """Client for interacting with Insight AI API"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.token = None
    
    def set_auth_token(self, token: str):
        """Set authentication token for API requests"""
        self.token = token
        self.session.headers.update({"Authorization": f"Bearer {token}"})
    
    def clear_auth_token(self):
        """Clear authentication token"""
        self.token = None
        self.session.headers.pop("Authorization", None)
    
    def register(self, name: str, email: str, password: str) -> Dict[str, Any]:
        """Register a new user"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/auth/register",
                json={"name": name, "email": email, "password": password}
            )
            return {"success": response.status_code == 201, "data": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def login(self, email: str, password: str) -> Dict[str, Any]:
        """Login user"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/auth/login",
                json={"email": email, "password": password}
            )
            if response.status_code == 200:
                data = response.json()
                # Handle nested token structure from backend
                access_token = data.get("tokens", {}).get("access_token") or data.get("access_token")
                if access_token:
                    self.set_auth_token(access_token)
                    return {"success": True, "data": data}
                else:
                    return {"success": False, "error": "No access token in response"}
            return {"success": False, "error": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def firebase_login(self, id_token: str) -> Dict[str, Any]:
        """Login with Firebase ID token"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/auth/firebase-login",
                json={"id_token": id_token}
            )
            if response.status_code == 200:
                data = response.json()
                # Handle nested token structure from backend
                access_token = data.get("tokens", {}).get("access_token") or data.get("access_token")
                if access_token:
                    self.set_auth_token(access_token)
                    return {"success": True, "data": data}
                else:
                    return {"success": False, "error": "No access token in response"}
            return {"success": False, "error": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def logout(self):
        """Logout user"""
        self.clear_auth_token()
    
    def get_current_user(self) -> Dict[str, Any]:
        """Get current user info"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/auth/me")
            return {"success": response.status_code == 200, "data": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def create_workspace(self, name: str, description: str = "") -> Dict[str, Any]:
        """Create a new workspace"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/workspaces/",
                json={"name": name, "description": description}
            )
            return {"success": response.status_code == 201, "data": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_workspaces(self) -> Dict[str, Any]:
        """Get user workspaces"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/workspaces/")
            return {"success": response.status_code == 200, "data": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def upload_document(self, workspace_id: str, file) -> Dict[str, Any]:
   
        try:
            
            files = {"file": (file.name, file, file.type)}

            
            upload_url = f"{self.base_url}/api/v1/workspaces/{workspace_id}/documents/upload"
            
            response = self.session.post(
                upload_url,
                files=files,
                timeout=60  
            )

            
            response.raise_for_status()

        
            if response.status_code == 202:
                return {"success": True, "data": response.json()}
            else:
                # This case is unlikely if raise_for_status() is used, but it's a safe fallback.
                return {
                    "success": False,
                    "error": f"Upload was accepted but with an unexpected status code: {response.status_code}",
                    "data": response.json()
                }

        except requests.exceptions.HTTPError as e:
        
            error_detail = "Unknown server error"
            try:
                error_detail = e.response.json().get("detail", str(e))
            except ValueError: # Catches cases where the response is not JSON
                error_detail = e.response.text
            return {"success": False, "error": f"Server Error: {error_detail}"}
        
        except requests.exceptions.RequestException as e:
            # Handle network-level errors (e.g., connection refused, timeout).
            return {"success": False, "error": f"Connection Error: {e}"}
        
        except Exception as e:
            # Catch any other unexpected errors during the process.
            return {"success": False, "error": f"An unexpected client-side error occurred: {str(e)}"}


        
    def get_workspace_documents(self, workspace_id: str) -> Dict[str, Any]:
        """Get documents in a workspace"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/workspaces/{workspace_id}/documents")
            return {"success": response.status_code == 200, "data": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_document_status(self, workspace_id: str, document_id: str) -> Dict[str, Any]:
        """Get detailed processing status for a document"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/workspaces/{workspace_id}/documents/{document_id}/status")
            return {"success": response.status_code == 200, "data": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def delete_workspace(self, workspace_id: str) -> Dict[str, Any]:
        """Delete a workspace"""
        try:
            response = self.session.delete(f"{self.base_url}/api/v1/workspaces/{workspace_id}")
            return {"success": response.status_code == 204, "message": "Workspace deleted successfully"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def delete_document(self, workspace_id: str, document_id: str) -> Dict[str, Any]:
        """Delete a document"""
        try:
            response = self.session.delete(f"{self.base_url}/api/v1/workspaces/{workspace_id}/documents/{document_id}")
            return {"success": response.status_code == 204, "message": "Document deleted successfully"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def query(self, workspace_id: str, question: str, session_id: str) -> Dict[str, Any]:
        """Send a query to the AI system"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/query/",
                json={
                    "workspace_id": workspace_id,
                    "query": question,
                    "session_id": session_id
                }
            )
            return {"success": response.status_code == 200, "data": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}


def ensure_client_initialized():
    """Ensure the client is initialized"""
    if st.session_state.client is None:
        st.session_state.client = InsightAIClient(API_BASE_URL)


def render_header():
    """Render the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>Insight AI</h1>
        <p>Advanced AI-powered document analysis and question answering</p>
    </div>
    """, unsafe_allow_html=True)


def render_login_page():
    """Render login/registration page with Firebase integration"""
    # Ensure client is initialized
    ensure_client_initialized()
    
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    
    # Show Firebase authentication only if properly configured
    if FIREBASE_ENABLED:
        # Firebase Authentication HTML
        firebase_config_js = json.dumps(FIREBASE_CONFIG)
        firebase_html = f"""
        <script src="https://www.gstatic.com/firebasejs/9.22.0/firebase-app-compat.js"></script>
        <script src="https://www.gstatic.com/firebasejs/9.22.0/firebase-auth-compat.js"></script>
        
        <div class="firebase-auth">
            <h3>🔐 Sign In</h3>
            
            <!-- Google Sign-In Button -->
            <button id="google-signin" class="google-signin-btn">
                🔍 Sign in with Google
            </button>
            
            <!-- Email/Password Form -->
            <div class="auth-divider">
                <span>or</span>
            </div>
            
            <form id="email-auth-form">
                <input type="email" id="email" placeholder="Email" required style="width: 100%; padding: 10px; margin: 5px 0; border: 1px solid #ddd; border-radius: 5px;">
                <input type="password" id="password" placeholder="Password" required style="width: 100%; padding: 10px; margin: 5px 0; border: 1px solid #ddd; border-radius: 5px;">
                <button type="submit" style="width: 100%; padding: 10px; margin: 10px 0; background: #667eea; color: white; border: none; border-radius: 5px; cursor: pointer;">Sign In</button>
            </form>
            
            <p style="text-align: center; margin: 10px 0;">
                Don't have an account? 
                <a href="#" id="show-register" style="color: #667eea;">Register here</a>
            </p>
            
            <!-- Registration Form (hidden by default) -->
            <form id="register-form" style="display: none;">
                <h4>Create Account</h4>
                <input type="text" id="reg-name" placeholder="Full Name" required style="width: 100%; padding: 10px; margin: 5px 0; border: 1px solid #ddd; border-radius: 5px;">
                <input type="email" id="reg-email" placeholder="Email" required style="width: 100%; padding: 10px; margin: 5px 0; border: 1px solid #ddd; border-radius: 5px;">
                <input type="password" id="reg-password" placeholder="Password" required style="width: 100%; padding: 10px; margin: 5px 0; border: 1px solid #ddd; border-radius: 5px;">
                <button type="submit" style="width: 100%; padding: 10px; margin: 10px 0; background: #28a745; color: white; border: none; border-radius: 5px; cursor: pointer;">Register</button>
                <p style="text-align: center;">
                    <a href="#" id="show-login" style="color: #667eea;">Back to Sign In</a>
                </p>
            </form>
            
            <div id="auth-message" style="margin: 10px 0;"></div>
        </div>
        
        <script>
        // Firebase configuration
        const firebaseConfig = {firebase_config_js};
        
        // Initialize Firebase
        firebase.initializeApp(firebaseConfig);
        const auth = firebase.auth();
        
        // Google provider
        const googleProvider = new firebase.auth.GoogleAuthProvider();
        
        // UI elements
        const emailForm = document.getElementById('email-auth-form');
        const registerForm = document.getElementById('register-form');
        const showRegister = document.getElementById('show-register');
        const showLogin = document.getElementById('show-login');
        const googleSignin = document.getElementById('google-signin');
        const messageDiv = document.getElementById('auth-message');
        
        // Show/hide forms
        showRegister.addEventListener('click', (e) => {{
            e.preventDefault();
            emailForm.style.display = 'none';
            registerForm.style.display = 'block';
        }});
        
        showLogin.addEventListener('click', (e) => {{
            e.preventDefault();
            emailForm.style.display = 'block';
            registerForm.style.display = 'none';
        }});
        
        // Helper function to show messages
        function showMessage(message, isError = false) {{
            messageDiv.innerHTML = `<div class="${{isError ? 'error-message' : 'success-message'}}">${{message}}</div>`;
        }}
        
        // Google Sign-In
        googleSignin.addEventListener('click', async () => {{
            try {{
                const result = await auth.signInWithPopup(googleProvider);
                const idToken = await result.user.getIdToken();
                
                // Send to Streamlit
                window.parent.postMessage({{
                    type: 'FIREBASE_AUTH_SUCCESS',
                    user: {{
                        uid: result.user.uid,
                        email: result.user.email,
                        displayName: result.user.displayName,
                        photoURL: result.user.photoURL
                    }},
                    idToken: idToken
                }}, '*');
                
                showMessage('Signed in successfully! Redirecting...');
            }} catch (error) {{
                showMessage(`Error: ${{error.message}}`, true);
            }}
        }});
        
        // Email/Password Sign-In
        emailForm.addEventListener('submit', async (e) => {{
            e.preventDefault();
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;
            
            try {{
                const result = await auth.signInWithEmailAndPassword(email, password);
                const idToken = await result.user.getIdToken();
                
                window.parent.postMessage({{
                    type: 'FIREBASE_AUTH_SUCCESS',
                    user: {{
                        uid: result.user.uid,
                        email: result.user.email,
                        displayName: result.user.displayName,
                        photoURL: result.user.photoURL
                    }},
                    idToken: idToken
                }}, '*');
                
                showMessage('Signed in successfully! Redirecting...');
            }} catch (error) {{
                showMessage(`Error: ${{error.message}}`, true);
            }}
        }});
        
        // Registration
        registerForm.addEventListener('submit', async (e) => {{
            e.preventDefault();
            const name = document.getElementById('reg-name').value;
            const email = document.getElementById('reg-email').value;
            const password = document.getElementById('reg-password').value;
            
            try {{
                const result = await auth.createUserWithEmailAndPassword(email, password);
                await result.user.updateProfile({{ displayName: name }});
                const idToken = await result.user.getIdToken();
                
                window.parent.postMessage({{
                    type: 'FIREBASE_AUTH_SUCCESS',
                    user: {{
                        uid: result.user.uid,
                        email: result.user.email,
                        displayName: name,
                        photoURL: result.user.photoURL
                    }},
                    idToken: idToken
                }}, '*');
                
                showMessage('Account created successfully! Redirecting...');
            }} catch (error) {{
                showMessage(`Error: ${{error.message}}`, true);
            }}
        }});
        
        // Listen for auth state changes
        auth.onAuthStateChanged((user) => {{
            if (user) {{
                console.log('User signed in:', user.email);
            }} else {{
                console.log('User signed out');
            }}
        }});
        </script>
        """
        
        # Render Firebase auth
        st.components.v1.html(firebase_html, height=500)
        
        # Handle Firebase authentication response
        if st.session_state.get("firebase_auth_pending"):
            if st.session_state.get("firebase_token"):
                with st.spinner("Authenticating with Firebase..."):
                    result = st.session_state.client.firebase_login(st.session_state.firebase_token)
                    if result["success"]:
                        st.session_state.authenticated = True
                        st.session_state.user = result["data"]["user"]
                        st.session_state.firebase_auth_pending = False
                        st.success("Firebase authentication successful!")
                        st.rerun()
                    else:
                        st.error(f"Firebase authentication failed: {result.get('error', 'Unknown error')}")
                        st.session_state.firebase_auth_pending = False
        
        # Add divider if Firebase is enabled
        st.markdown('<div class="auth-divider"><span>or use developer login</span></div>', unsafe_allow_html=True)
    else:
        # Show message about Firebase configuration
        st.info("🔧 Firebase authentication is not configured. Using developer login for testing.")
    
    # Developer login section (always shown)
    st.markdown('<div class="developer-login">', unsafe_allow_html=True)
    st.markdown("### 🛠️ Developer Login")
    st.markdown("*Quick login for development and testing*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Login as Admin", use_container_width=True):
            with st.spinner("Logging in..."):
                result = st.session_state.client.login("admin@example.com", "admin123")
                if result["success"]:
                    st.session_state.authenticated = True
                    st.session_state.user = result["data"]["user"]
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error(f"Login failed: {result.get('error', 'Unknown error')}")
    
    with col2:
        if st.button("Login as User", use_container_width=True):
            with st.spinner("Logging in..."):
                result = st.session_state.client.login("user@example.com", "user123")
                if result["success"]:
                    st.session_state.authenticated = True
                    st.session_state.user = result["data"]["user"]
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error(f"Login failed: {result.get('error', 'Unknown error')}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def check_authentication():
    """Check if user is authenticated"""
    if not st.session_state.get("authenticated", False):
        return False
    
    if not st.session_state.get("user"):
        return False
    
    # Verify token is still valid
    if hasattr(st.session_state.client, 'token') and st.session_state.client.token:
        result = st.session_state.client.get_current_user()
        if not result["success"]:
            # Token expired, clear auth
            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.client.clear_auth_token()
            return False
    
    return True


def render_sidebar():
    """Render the sidebar with user info and workspaces"""
    
    # Add sidebar state management
    if "sidebar_collapsed" not in st.session_state:
        st.session_state.sidebar_collapsed = False
    
    # Sidebar toggle button
    col1, col2 = st.columns([1, 10])
    with col1:
        if st.button("☰" if st.session_state.sidebar_collapsed else "✖", 
                    key="sidebar_toggle",
                    help="Toggle sidebar"):
            st.session_state.sidebar_collapsed = not st.session_state.sidebar_collapsed
            st.rerun()
    
    # Only show sidebar content if not collapsed
    if not st.session_state.sidebar_collapsed:
        with st.sidebar:
            # Enhanced header
            st.markdown("""
            <div style='text-align: center; padding: 20px 0;'>
                <h1 style='color: #007bff; margin: 0;'>🤖 Insight AI</h1>
                <p style='color: #6c757d; margin: 5px 0;'>Intelligent Document Analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            
            # User info
            if st.session_state.get("user"):
                user = st.session_state.user
                st.markdown("### 👤 User Profile")
                
                # User info card
                st.markdown(f"""
                <div style='background: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0;'>
                    <h4 style='margin: 0; color: #495057;'>{user['name']}</h4>
                    <p style='margin: 5px 0; color: #6c757d;'>📧 {user['email']}</p>
                    <p style='margin: 5px 0; color: #6c757d;'>🆔 {user.get('user_id', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("🚪 Logout", use_container_width=True, type="secondary"):
                    st.session_state.client.logout()
                    st.session_state.authenticated = False
                    st.session_state.user = None
                    st.session_state.current_workspace = None
                    st.session_state.workspaces = []
                    st.session_state.messages = []
                    st.success("Logged out successfully!")
                    st.rerun()
            
            st.divider()
            
            # Enhanced Workspace management section
            st.markdown("### 🏢 Workspace Management")
            
            # Load workspaces
            if not st.session_state.workspaces:
                with st.spinner("Loading workspaces..."):
                    result = st.session_state.client.get_workspaces()
                    if result["success"]:
                        st.session_state.workspaces = result["data"]
            
            # Current workspace info
            if st.session_state.current_workspace:
                workspace = st.session_state.current_workspace
                st.markdown(f"""
                <div style='background: #e3f2fd; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #2196f3;'>
                    <h4 style='margin: 0; color: #1976d2;'>📁 {workspace['name']}</h4>
                    <p style='margin: 5px 0; color: #424242; font-size: 0.9em;'>{workspace.get('description', 'No description')}</p>
                    <div style='display: flex; justify-content: space-between; margin-top: 10px;'>
                        <span style='color: #666; font-size: 0.8em;'>📄 {workspace.get('document_count', 0)} docs</span>
                        <span style='color: #666; font-size: 0.8em;'>🧩 {workspace.get('total_chunks', 0)} chunks</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Workspace selection
            workspace_names = ["Select a workspace..."] + [ws["name"] for ws in st.session_state.workspaces]
            current_name = st.session_state.current_workspace["name"] if st.session_state.current_workspace else "Select a workspace..."
            
            selected = st.selectbox(
                "🔄 Switch workspace:",
                workspace_names,
                index=workspace_names.index(current_name) if current_name in workspace_names else 0,
                help="Choose a different workspace to work with"
            )
        
            if selected != "Select a workspace..." and selected != current_name:
                st.session_state.current_workspace = next(
                    ws for ws in st.session_state.workspaces if ws["name"] == selected
                )
                st.session_state.messages = []  # Clear messages when switching workspace
                st.rerun()
            
            # Enhanced Workspace management options
            if st.session_state.current_workspace:
                st.markdown("#### ⚙️ Workspace Actions")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔄 Refresh", use_container_width=True, help="Refresh workspace data"):
                        # Refresh workspaces
                        with st.spinner("Refreshing..."):
                            result = st.session_state.client.get_workspaces()
                            if result["success"]:
                                st.session_state.workspaces = result["data"]
                        st.rerun()
                
                with col2:
                    if st.button("🗑️ Delete", use_container_width=True, type="secondary", help="Delete this workspace"):
                        # Show confirmation dialog
                        if st.session_state.get("confirm_delete_workspace", False):
                            # User confirmed deletion
                            with st.spinner("Deleting workspace..."):
                                result = st.session_state.client.delete_workspace(
                                    st.session_state.current_workspace["workspace_id"]
                                )
                                if result["success"]:
                                    st.success("Workspace deleted successfully!")
                                    # Clear current workspace and refresh list
                                    st.session_state.current_workspace = None
                                    st.session_state.workspaces = []
                                    st.session_state.messages = []
                                    st.session_state.confirm_delete_workspace = False
                                    st.rerun()
                                else:
                                    st.error(f"Failed to delete workspace: {result.get('error', 'Unknown error')}")
                                    st.session_state.confirm_delete_workspace = False
                        else:
                            # Show confirmation
                            st.session_state.confirm_delete_workspace = True
                            st.rerun()
                
                # Show confirmation dialog if needed
                if st.session_state.get("confirm_delete_workspace", False):
                    st.warning(f"⚠️ **Are you sure you want to delete '{st.session_state.current_workspace['name']}'?**")
                    st.warning("This action cannot be undone and will delete all documents in this workspace.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Yes, Delete", type="primary", use_container_width=True):
                            # This will trigger the deletion on next rerun
                            pass
                    with col2:
                        if st.button("❌ Cancel", use_container_width=True):
                            st.session_state.confirm_delete_workspace = False
                            st.rerun()
            
            # Create new workspace
            with st.expander("➕ Create New Workspace"):
                new_name = st.text_input("Workspace Name")
                new_desc = st.text_area("Description (optional)")
                
                if st.button("Create", use_container_width=True):
                    if new_name:
                        with st.spinner("Creating workspace..."):
                            result = st.session_state.client.create_workspace(new_name, new_desc)
                            if result["success"]:
                                st.session_state.workspaces.append(result["data"])
                                st.session_state.current_workspace = result["data"]
                                st.success("Workspace created!")
                                st.rerun()
                            else:
                                st.error(f"Failed to create workspace: {result.get('error', 'Unknown error')}")
                    else:
                        st.error("Please enter a workspace name")
            
            # Enhanced Document upload section (only show if workspace is selected)
            if st.session_state.current_workspace:
                st.divider()
                st.markdown("### 📁 Document Management")
                
                # Enhanced Upload new document section
                with st.expander("📤 Upload New Document", expanded=False):
                    st.markdown("**Drag and drop or click to browse**")
                    uploaded_file = st.file_uploader(
                        "Choose a file",
                        type=['pdf', 'docx', 'doc', 'txt', 'md', 'png', 'jpg', 'jpeg'],
                        help="Supported: PDF, Word, Text, Markdown, Images",
                        label_visibility="collapsed"
                    )
                    
                    if uploaded_file is not None:
                        # File preview
                        st.markdown(f"""
                        <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;'>
                            <h5 style='margin: 0; color: #495057;'>📄 {uploaded_file.name}</h5>
                            <div style='display: flex; justify-content: space-between; margin-top: 8px;'>
                                <span style='color: #6c757d; font-size: 0.9em;'>💾 {uploaded_file.size / 1024:.1f} KB</span>
                                <span style='color: #6c757d; font-size: 0.9em;'>🏷️ {uploaded_file.type}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button("🚀 Upload & Process", use_container_width=True, type="primary"):
                            with st.spinner("Uploading and processing document..."):
                                result = st.session_state.client.upload_document(
                                    st.session_state.current_workspace["workspace_id"],
                                    uploaded_file
                                )
                                if result["success"]:
                                    st.success("✅ Document uploaded! Processing started.")
                                    st.balloons()
                                    st.rerun()
                                else:
                                    st.error(f"❌ Upload failed: {result.get('error', 'Unknown error')}")
                
                # Document list header with refresh
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown("#### 📚 Document Library")
                with col2:
                    if st.button("🔄", help="Refresh documents", use_container_width=True):
                        st.rerun()
                
                # Load and display documents
                with st.spinner("Loading documents..."):
                    docs_result = st.session_state.client.get_workspace_documents(
                        st.session_state.current_workspace["workspace_id"]
                    )
                    
                    # Check if any documents are still processing for auto-refresh
                    has_processing_docs = False
                    if docs_result["success"]:
                        documents = docs_result["data"]
                        has_processing_docs = any(doc.get("status") == "processing" for doc in documents)
                    
                    # Auto-refresh every 10 seconds if there are processing documents
                    if has_processing_docs:
                        st.info("⏳ **Documents are processing** - Auto-refreshing every 10 seconds...")
                        import time
                        time.sleep(0.1)  # Small delay to prevent too frequent refreshes
                        # Add a rerun after 10 seconds using JavaScript
                        st.markdown("""
                        <script>
                        setTimeout(function() {
                            window.location.reload();
                        }, 10000);
                        </script>
                        """, unsafe_allow_html=True)
                    
                    if docs_result["success"]:
                        documents = docs_result["data"]
                        if documents:
                            for doc in documents:
                                status_emoji = {
                                    "processing": "⏳",
                                    "completed": "✅", 
                                    "ready": "✅",
                                    "failed": "❌",
                                    "error": "❌",
                                    "pending": "🔄"
                                }.get(doc.get("status", "unknown"), "❓")
                                
                                # Enhanced document card
                                status_color = {
                                    "processing": "#007bff",
                                    "completed": "#28a745", 
                                    "ready": "#28a745",
                                    "failed": "#dc3545",
                                    "error": "#dc3545",
                                    "pending": "#ffc107"
                                }.get(doc.get("status", "unknown"), "#6c757d")
                                
                                st.markdown(f"""
                                <div class='document-container' style='border-left: 4px solid {status_color};'>
                                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                                        <div style='flex: 1;'>
                                            <h5 style='margin: 0; color: #495057;'>{status_emoji} {doc['file_name']}</h5>
                                            <div style='margin: 5px 0; font-size: 0.85em; color: #6c757d;'>
                                                💾 {doc.get('file_size', 0) / 1024:.1f} KB • 
                                                🏷️ {doc.get('file_type', 'Unknown').split('/')[-1].upper()} • 
                                                📅 {doc.get('created_at', 'Unknown')[:10] if doc.get('created_at') else 'Unknown'}
                                            </div>
                                            <span style='background: {status_color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em;'>
                                                {doc.get("status", "unknown").title()}
                                            </span>
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Action buttons row
                                col1, col2, col3 = st.columns([1, 1, 1])
                                
                                with col1:
                                    # Show processing details button for processing documents
                                    if doc.get("status") in ["processing", "error"]:
                                        if st.button("📊 Details", key=f"details_{doc.get('document_id', 'unknown')}", use_container_width=True, type="secondary"):
                                            # Get detailed status
                                            status_result = st.session_state.client.get_document_status(
                                                st.session_state.current_workspace["workspace_id"],
                                                doc["document_id"]
                                            )
                                            
                                            if status_result["success"]:
                                                status_data = status_result["data"]
                                                
                                                # Show processing progress in an expander
                                                with st.expander(f"🔍 Processing Details - {doc['file_name']}", expanded=True):
                                                    # Progress bar
                                                    progress = status_data.get("progress_percentage", 0)
                                                    st.progress(progress / 100)
                                                    st.write(f"**Progress: {progress}%**")
                                                    
                                                    # Processing steps
                                                    st.write("**Processing Steps:**")
                                                    for step in status_data.get("steps", []):
                                                        step_status = step["status"]
                                                        step_emoji = {
                                                            "completed": "✅",
                                                            "in_progress": "⏳", 
                                                            "pending": "⏸️",
                                                            "skipped": "⏭️",
                                                            "failed": "❌"
                                                        }.get(step_status, "❓")
                                                        
                                                        # In E:\insight_ai\frontend\app.py
                                                        st.write(f"{step_emoji} **{step.get('action', 'Unknown Step').replace('_', ' ').title()}** - {step_status.replace('_', ' ').title()}")
                                                        st.caption(step["description"])
                                                    
                                                    # Additional info
                                                    col_a, col_b = st.columns(2)
                                                    with col_a:
                                                        st.metric("Chunks Created", status_data.get("total_chunks", 0))
                                                    with col_b:
                                                        st.metric("OCR Applied", "Yes" if status_data.get("ocr_applied") else "No")
                                                    
                                                    # Error message if any
                                                    if status_data.get("error_message"):
                                                        st.error(f"**Error:** {status_data['error_message']}")
                                                    
                                                    # Refresh button
                                                    if st.button("🔄 Refresh Status", key=f"refresh_{doc['document_id']}"):
                                                        st.rerun()
                                            else:
                                                st.error("Failed to get processing details")
                                
                                with col2:
                                    # View/Download button for ready documents  
                                    if doc.get("status") == "ready":
                                        if st.button("👁️ View", key=f"view_{doc.get('document_id', 'unknown')}", use_container_width=True):
                                            st.info("Document viewing feature coming soon!")
                                
                                with col3:
                                    # Delete document button
                                    doc_id = doc.get('document_id', 'unknown')
                                    confirm_key = f"confirm_delete_doc_{doc_id}"
                                    
                                    if st.session_state.get(confirm_key, False):
                                        # Show confirmation
                                        if st.button("✅", key=f"confirm_{doc_id}", use_container_width=True, type="primary", help="Confirm deletion"):
                                            # Delete the document
                                            with st.spinner("Deleting..."):
                                                result = st.session_state.client.delete_document(
                                                    st.session_state.current_workspace["workspace_id"],
                                                    doc_id
                                                )
                                                if result["success"]:
                                                    st.success("Document deleted!")
                                                    st.session_state[confirm_key] = False
                                                    st.rerun()
                                                else:
                                                    st.error(f"Failed to delete: {result.get('error', 'Unknown error')}")
                                                    st.session_state[confirm_key] = False
                                    else:
                                        if st.button("🗑️", key=f"delete_{doc_id}", use_container_width=True, type="secondary", help="Delete document"):
                                            st.session_state[confirm_key] = True
                                            st.rerun()
                                
                                # Show confirmation warning if delete is pending
                                if st.session_state.get(f"confirm_delete_doc_{doc.get('document_id', 'unknown')}", False):
                                    st.warning(f"⚠️ Delete '{doc['file_name']}'? This cannot be undone.")
                                    if st.button("❌ Cancel", key=f"cancel_{doc.get('document_id', 'unknown')}", type="secondary"):
                                        st.session_state[f"confirm_delete_doc_{doc.get('document_id', 'unknown')}"] = False
                                        st.rerun()
                                
                                st.markdown("<br>", unsafe_allow_html=True)  # Add spacing between documents
                        else:
                            # Enhanced empty state
                            st.markdown("""
                            <div style='text-align: center; padding: 30px; background: #f8f9fa; border-radius: 10px; margin: 20px 0;'>
                                <h3 style='color: #6c757d; margin: 0;'>📄 No Documents Yet</h3>
                                <p style='color: #868e96; margin: 10px 0;'>Upload your first document to start analyzing with AI</p>
                                <p style='color: #adb5bd; font-size: 0.9em;'>Supported: PDF, Word, Text, Images</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error("❌ Failed to load documents. Please refresh and try again.")
                
                # Sidebar footer with stats
                if st.session_state.current_workspace:
                    st.divider()
                    workspace = st.session_state.current_workspace
                    st.markdown("#### 📊 Workspace Stats")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Documents", workspace.get('document_count', 0), help="Total documents in workspace")
                    with col2:
                        st.metric("Chunks", workspace.get('total_chunks', 0), help="Total processed chunks")
    
    # If sidebar is collapsed, show minimal interface
    elif st.session_state.sidebar_collapsed:
        # Collapsed sidebar indicator
        st.markdown("""
        <div style='position: fixed; top: 70px; left: 10px; z-index: 1000; background: white; 
                    border: 1px solid #dee2e6; border-radius: 5px; padding: 5px;'>
            <span style='color: #6c757d; font-size: 0.8em;'>Sidebar Hidden</span>
        </div>
        """, unsafe_allow_html=True)


def render_chat_interface():
    """Render the main chat interface"""
    if not st.session_state.current_workspace:
        st.info("👋 Please select or create a workspace to start chatting!")
        return
    
    st.markdown(f"### 💬 Chat - {st.session_state.current_workspace['name']}")
    
    # Check if workspace has documents
    docs_result = st.session_state.client.get_workspace_documents(
        st.session_state.current_workspace["workspace_id"]
    )
    
    has_documents = False
    if docs_result["success"]:
        documents = docs_result["data"]
        has_documents = len(documents) > 0
    
    # Show helpful message if no documents
    if not has_documents:
        st.info("📄 **No documents uploaded yet!** Upload documents in the sidebar to start analyzing them with AI.")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    chat_placeholder = "Upload documents first, then ask me anything about them..." if not has_documents else "Ask me anything about your documents..."
    
    if prompt := st.chat_input(chat_placeholder, disabled=not has_documents):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.client.query(
                    st.session_state.current_workspace["workspace_id"],
                    prompt,
                    st.session_state.session_id
                )
                
                if result["success"]:
                    response = result["data"]["response"]
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    error_msg = f"Sorry, I encountered an error: {result.get('error', 'Unknown error')}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


def main():
    """Main application function"""
    # Initialize session state FIRST
    init_session_state()
    
    # Ensure client is initialized
    ensure_client_initialized()
    
    # Check authentication
    if not check_authentication():
        render_login_page()
        return
    
    # Render authenticated interface
    render_header()
    render_sidebar()
    render_chat_interface()


if __name__ == "__main__":
    main()
