import React, { useState, useEffect } from 'react';
import Layout from '@theme/Layout';
import { BetterAuthProvider, useBetterAuth } from '../../components/auth/BetterAuthProvider';

// Inner component that uses the auth context
function SigninPageContent() {
  const [SigninForm, setSigninForm] = useState(null);
  const [error, setError] = useState(null);
  const { signin } = useBetterAuth();

  useEffect(() => {
    // Dynamically import the SigninForm component after component mounts (client-side only)
    import('@site/src/components/auth/SigninForm').then((module) => {
      setSigninForm(() => module.default);
    }).catch(err => {
      console.error('Error loading SigninForm:', err);
    });
  }, []);

  const handleSignin = async (credentials) => {
    try {
      // Use Better Auth signin function
      const response = await signin(credentials);

      if (!response) {
        throw new Error('Signin failed');
      }

      // Wait a brief moment to ensure auth state updates, then redirect to home page
      setTimeout(() => {
        window.location.href = '/';
      }, 500);

      return response;
    } catch (err) {
      console.error('Signin error:', err);

      // Handle different types of errors
      if (err.message.includes('network') || err.message.includes('fetch')) {
        setError('Network error: Unable to connect to the server. Please check your internet connection and try again.');
      } else if (err.message.includes('401') || err.message.toLowerCase().includes('invalid') || err.message.toLowerCase().includes('incorrect')) {
        setError('Invalid credentials: Please check your email and password.');
      } else if (err.message.includes('400') || err.message.toLowerCase().includes('invalid')) {
        setError(`Invalid input: ${err.message}`);
      } else if (err.message.includes('500')) {
        setError('Server error: Please try again later.');
      } else {
        setError(err.message || 'An unexpected error occurred during signin. Please try again.');
      }

      throw err;
    }
  };

  return (
    <Layout title="Sign In" description="Sign in to your account for personalized learning experience">
      <div className="min-h-screen bg-gradient-to-br from-green-50 via-emerald-50 to-teal-50 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
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
          {SigninForm ? (
            <SigninForm
              onSignin={handleSignin}
              onError={(msg) => console.error('Signin error:', msg)}
            />
          ) : (
            <div className="text-center bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
              <div className="loading-spinner inline-block w-8 h-8 border-4 border-gray-200 border-t-green-500 rounded-full animate-spin"></div>
              <p className="mt-4 text-gray-600">Loading signin form...</p>
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
}

// Main component that wraps content with auth provider
function SigninPage() {
  return (
    <BetterAuthProvider>
      <SigninPageContent />
    </BetterAuthProvider>
  );
}

export default SigninPage;