import React, { createContext, useContext, useState, useEffect } from 'react';

// Create Auth Context
const AuthContext = createContext();

// Auth Provider Component
export const BetterAuthProvider = ({ children, backendUrl = 'http://localhost:8002' }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);


  // Check for stored auth data on initial load
  useEffect(() => {
    // Only run on client side
    if (typeof window === 'undefined') {
      setLoading(false);
      return;
    }

    const storedToken = localStorage.getItem('auth_token');
    const storedUser = localStorage.getItem('auth_user');

    if (storedToken && storedUser) {
      try {
        const parsedUser = JSON.parse(storedUser);
        setToken(storedToken);
        setUser(parsedUser);
        setIsAuthenticated(true);
      } catch (error) {
        console.error('Error parsing stored auth data:', error);
        // Clear invalid stored data
        localStorage.removeItem('auth_token');
        localStorage.removeItem('auth_user');
      }
    }

    setLoading(false);
  }, []);

  // Signup function using backend API
  const signup = async (userData) => {
    const url = `${backendUrl}/api/auth/signup`;
    console.log('Signup request URL:', url);
    console.log('Signup request data:', userData);
    
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(userData),
    });

    console.log('Signup response status:', response.status, response.statusText);
    console.log('Signup response URL:', response.url);

    if (!response.ok) {
      let errorMessage = `Signup failed (${response.status} ${response.statusText})`;
      try {
        const errorData = await response.json();
        console.log('Signup error data:', errorData);
        errorMessage = errorData.detail || errorData.message || errorMessage;
      } catch (e) {
        console.error('Failed to parse error response:', e);
        const text = await response.text().catch(() => '');
        if (text) {
          errorMessage = `${errorMessage}: ${text}`;
        }
      }
      throw new Error(errorMessage);
    }

    const data = await response.json();
    
    console.log('Signup API response:', data);

    // Extract token and user from the response structure
    // Backend returns: { user: {...}, session: { token: "...", ... } }
    const token = data.session?.token || data.token;
    const user = data.user || data;

    if (!token) {
      console.error('No token in response:', data);
      throw new Error('No token received from server');
    }

    if (!user) {
      console.error('No user in response:', data);
      throw new Error('No user data received from server');
    }

    // Store the token and user info in localStorage
    localStorage.setItem('auth_token', token);
    localStorage.setItem('auth_user', JSON.stringify(user));

    setToken(token);
    setUser(user);
    setIsAuthenticated(true);

    console.log('Auth state updated successfully');
    return data;
  };

  // Signin function using backend API
  const signin = async (credentials) => {
    const response = await fetch(`${backendUrl}/api/auth/signin`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(credentials),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Signin failed');
    }

    const data = await response.json();

    // Store the token and user info in localStorage
    localStorage.setItem('auth_token', data.token);
    localStorage.setItem('auth_user', JSON.stringify(data.user));

    setToken(data.token);
    setUser(data.user);
    setIsAuthenticated(true);

    return data;
  };

  // Signout function
  const signout = () => {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('auth_user');
    setToken(null);
    setUser(null);
    setIsAuthenticated(false);
  };

  // Get user profile
  const getProfile = async () => {
    if (!token) {
      throw new Error('Not authenticated');
    }

    const response = await fetch(`${backendUrl}/api/auth/profile`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to fetch profile');
    }

    const data = await response.json();
    setUser(data);
    // Update localStorage with new user data
    localStorage.setItem('auth_user', JSON.stringify(data));
    return data;
  };

  // Update profile function
  const updateProfile = async (profileData) => {
    if (!token) {
      throw new Error('Not authenticated');
    }

    const response = await fetch(`${backendUrl}/api/auth/profile`, {
      method: 'PUT',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(profileData),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to update profile');
    }

    const data = await response.json();
    setUser(data);
    return data;
  };

  const value = {
    user,
    token,
    isAuthenticated,
    loading,
    signup,
    signin,
    signout,
    getProfile,
    updateProfile
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

// Custom hook to use auth context
export const useBetterAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useBetterAuth must be used within a BetterAuthProvider');
  }
  return context;
};