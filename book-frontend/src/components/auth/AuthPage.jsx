import React, { useState } from 'react';
import { useBetterAuth } from './BetterAuthProvider';
import SigninForm from './SigninForm';
import SignupForm from './SignupForm';
import './auth-styles.css'; // Import our auth-specific styles

const AuthPage = ({ mode = 'signin' }) => {
  const [currentMode, setCurrentMode] = useState(mode);
  const { signin, signup, onError } = useBetterAuth();
  const [authError, setAuthError] = useState('');

  const handleSignin = async (credentials) => {
    try {
      setAuthError('');
      await signin(credentials);
    } catch (error) {
      setAuthError(error.message);
    }
  };

  const handleSignup = async (userData) => {
    try {
      setAuthError('');
      await signup(userData);
    } catch (error) {
      setAuthError(error.message);
    }
  };

  const toggleMode = () => {
    setCurrentMode(currentMode === 'signin' ? 'signup' : 'signin');
    setAuthError('');
  };

  return (
    <div className="container margin-vert--lg">
      <div className="row">
        <div className="col col--6 col--offset-3">
          {authError && (
            <div className="alert alert--danger margin-bottom--md">
              {authError}
            </div>
          )}

          {currentMode === 'signin' ? (
            <SigninForm onSignin={handleSignin} onError={setAuthError} />
          ) : (
            <SignupForm onSignup={handleSignup} onError={setAuthError} />
          )}

          <div className="text--center margin-top--lg">
            <p>
              {currentMode === 'signin'
                ? "Don't have an account?"
                : "Already have an account?"}
              <button
                onClick={toggleMode}
                className="button button--link"
                style={{ marginLeft: '0.5rem' }}
              >
                {currentMode === 'signin' ? 'Sign up' : 'Sign in'}
              </button>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AuthPage;