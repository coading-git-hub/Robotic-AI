import React, { useState, useEffect } from 'react';
import Layout from '@theme/Layout';
import { BetterAuthProvider, useBetterAuth } from '../../components/auth/BetterAuthProvider';

// Inner component that uses the auth context
function ProfilePageContent() {
  const [ProfileForm, setProfileForm] = useState(null);
  const [error, setError] = useState(null);
  const {
    user,
    isAuthenticated,
    updateProfile,
    getProfile,
    signout
  } = useBetterAuth();

  useEffect(() => {
    // Dynamically import the ProfileForm component after component mounts (client-side only)
    import('@site/src/components/auth/ProfileForm').then((module) => {
      setProfileForm(() => module.default);
    }).catch(err => {
      console.error('Error loading ProfileForm:', err);
    });
  }, []);

  const handleSignOut = () => {
    signout();
    window.location.href = '/';
  };

  if (!isAuthenticated) {
    return (
      <Layout title="Profile" description="Please sign in to view your profile">
        <div className="container margin-vert--lg">
          <div className="row">
            <div className="col col--6 col--offset-3">
              <div className="alert alert--warning margin-bottom--md">
                <p>You must be signed in to access your profile.</p>
              </div>
              <div className="text--center">
                <a href="/auth/signin" className="button button--primary">
                  Sign In
                </a>
              </div>
            </div>
          </div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout title="Your Profile" description="Manage your profile and learning preferences">
      <div className="container margin-vert--lg">
        <div className="row">
          <div className="col col--8 col--offset-2">
            {error && (
              <div className="alert alert--danger margin-bottom--md">
                <p>{error}</p>
                <button
                  onClick={() => setError(null)}
                  className="button button--sm button--outline button--secondary"
                >
                  Dismiss
                </button>
              </div>
            )}

            <div className="margin-bottom--lg text-center">
              <h1 className="text-3xl font-bold text-gray-800">Your Profile</h1>
              <p className="text-gray-600">Manage your personal information and learning preferences</p>
            </div>

            {ProfileForm ? (
              <ProfileForm
                onProfileUpdateSuccess={(updatedUser) => {
                  console.log('Profile updated successfully:', updatedUser);
                  // Redirect to home page after successful update
                  window.location.href = '/';
                }}
                onProfileUpdateError={(errorMessage) => {
                  console.error('Profile update error:', errorMessage);
                }}
              />
            ) : (
              <div className="text--center padding-vert--md">
                <div className="loading-spinner" style={{ display: 'inline-block', width: '20px', height: '20px', border: '3px solid #f3f3f3', borderTop: '3px solid #3498db', borderRadius: '50%', animation: 'spin 1s linear infinite' }}></div>
                <p className="margin-top--sm">Loading profile form...</p>
              </div>
            )}

            <div className="margin-top--lg text-center">
              <button
                onClick={handleSignOut}
                className="button button--outline button--danger"
              >
                Sign Out
              </button>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}

// Main component that wraps content with auth provider
function ProfilePage() {
  return (
    <BetterAuthProvider>
      <ProfilePageContent />
    </BetterAuthProvider>
  );
}

export default ProfilePage;