import React, { useState, useEffect } from 'react';
import Layout from '@theme/Layout';
import { BetterAuthProvider, useBetterAuth } from '../../components/auth/BetterAuthProvider';

// Inner component that uses the auth context
function SignupPageContent() {
  const [SignupForm, setSignupForm] = useState(null);
  const [error, setError] = useState(null);
  const { signup } = useBetterAuth();

  useEffect(() => {
    // Dynamically import the SignupForm component after component mounts (client-side only)
    import('@site/src/components/auth/SignupForm').then((module) => {
      setSignupForm(() => module.default);
    }).catch(err => {
      console.error('Error loading SignupForm:', err);
    });
  }, []);

  const handleSignup = async (userData) => {
    try {
      console.log('Starting signup process...', userData);
      // Use Better Auth signup function
      const response = await signup(userData);

      console.log('Signup response received:', response);

      if (!response) {
        throw new Error('Signup failed - no response received');
      }

      // Verify we have the necessary data
      if (!response.user && !response.session) {
        console.warn('Signup response missing user or session data:', response);
      }

      console.log('Signup successful, redirecting to home page...');
      // Wait a brief moment to ensure auth state updates, then redirect to home page
      setTimeout(() => {
        console.log('Redirecting to /');
        window.location.href = '/';
      }, 500);

      return response;
    } catch (err) {
      console.error('Signup error:', err);

      // Handle different types of errors
      if (err.message.includes('network') || err.message.includes('fetch')) {
        setError('Network error: Unable to connect to the server. Please check your internet connection and try again.');
      } else if (err.message.includes('409') || err.message.toLowerCase().includes('exists')) {
        setError('Account already exists with this email. Please try signing in instead.');
      } else if (err.message.includes('400') || err.message.toLowerCase().includes('invalid')) {
        setError(`Invalid input: ${err.message}`);
      } else if (err.message.includes('500')) {
        setError('Server error: Please try again later.');
      } else {
        setError(err.message || 'An unexpected error occurred during signup. Please try again.');
      }

      throw err;
    }
  };

  return (
    <Layout title="Sign Up" description="Create your account for personalized learning experience">
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
        <div className="w-full max-w-lg">
          {error && (
            <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4 flex items-center justify-between">
              <p className="text-red-800 text-sm">{error}</p>
              <button
                onClick={() => setError(null)}
                className="text-red-600 hover:text-red-800 font-medium text-sm transition-colors duration-200"
              >
                ×
              </button>
            </div>
          )}
          {SignupForm ? (
            <SignupForm
              onSignup={handleSignup}
              onError={(msg) => console.error('Signup error:', msg)}
            />
          ) : (
            <div className="text-center bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
              <div className="loading-spinner inline-block w-8 h-8 border-4 border-gray-200 border-t-blue-500 rounded-full animate-spin"></div>
              <p className="mt-4 text-gray-600">Loading signup form...</p>
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
}

// Main component that wraps content with auth provider
function SignupPage() {
  return (
    <BetterAuthProvider>
      <SignupPageContent />
    </BetterAuthProvider>
  );
}

export default SignupPage;